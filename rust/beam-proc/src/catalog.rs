use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub struct Catalog {
    nodes: Vec<CatalogNode>,
}

#[derive(Debug, Default)]
struct CatalogNode {
    children: HashMap<i32, usize>,
    sorted_child_ids: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct CatalogPrefix<'a> {
    node: Option<&'a CatalogNode>,
}

impl<'a> CatalogPrefix<'a> {
    pub fn valid_next_sorted(self) -> &'a [i32] {
        self.node
            .map(|node| node.sorted_child_ids.as_slice())
            .unwrap_or_default()
    }
}

impl Catalog {
    pub fn load_json(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let entries = serde_json::from_str::<Vec<Vec<CatalogToken>>>(&data)?;
        let mut catalog = Self {
            nodes: vec![CatalogNode::default()],
        };
        for entry in entries {
            catalog.insert(entry.into_iter().map(CatalogToken::into_i32));
        }
        catalog.finalize();
        Ok(catalog)
    }

    fn finalize(&mut self) {
        for node in &mut self.nodes {
            node.sorted_child_ids = node.children.keys().copied().collect();
            node.sorted_child_ids.sort_unstable();
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[cfg(test)]
    fn contains_next(
        &self,
        parent_token_ids: &[i32],
        prompt_token_count: Option<usize>,
        token_id: i32,
    ) -> bool {
        self.prefix(parent_token_ids, prompt_token_count)
            .node
            .is_some_and(|node| node.children.contains_key(&token_id))
    }

    pub fn prefix(
        &self,
        parent_token_ids: &[i32],
        prompt_token_count: Option<usize>,
    ) -> CatalogPrefix<'_> {
        let prefix = self.catalog_prefix(parent_token_ids, prompt_token_count);
        let mut node_index = 0usize;
        for token_id in prefix {
            let Some(next_index) = self.nodes[node_index].children.get(token_id) else {
                return CatalogPrefix { node: None };
            };
            node_index = *next_index;
        }
        CatalogPrefix {
            node: self.nodes.get(node_index),
        }
    }

    fn insert(&mut self, token_ids: impl IntoIterator<Item = i32>) {
        let mut node_index = 0usize;
        for token_id in token_ids {
            if let Some(next_index) = self.nodes[node_index].children.get(&token_id).copied() {
                node_index = next_index;
                continue;
            }
            let next_index = self.nodes.len();
            self.nodes.push(CatalogNode::default());
            self.nodes[node_index].children.insert(token_id, next_index);
            node_index = next_index;
        }
    }

    fn catalog_prefix<'a>(
        &'a self,
        parent_token_ids: &'a [i32],
        prompt_token_count: Option<usize>,
    ) -> &'a [i32] {
        if let Some(prompt_token_count) = prompt_token_count {
            return parent_token_ids
                .get(prompt_token_count..)
                .unwrap_or_default();
        }

        parent_token_ids
            .iter()
            .position(|token_id| self.nodes[0].children.contains_key(token_id))
            .and_then(|index| parent_token_ids.get(index..))
            .unwrap_or_default()
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self {
            nodes: vec![CatalogNode::default()],
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CatalogToken {
    I32(i32),
    I64(i64),
    U64(u64),
}

impl CatalogToken {
    fn into_i32(self) -> i32 {
        match self {
            Self::I32(value) => value,
            Self::I64(value) => value as i32,
            Self::U64(value) => value as i32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Catalog;

    #[test]
    fn contains_next_uses_prompt_token_count_suffix() {
        let mut catalog = Catalog::default();
        catalog.insert([10, 20, 30]);
        catalog.insert([10, 20, 40]);

        assert!(catalog.contains_next(&[1, 2, 10, 20], Some(2), 30));
        assert!(catalog.contains_next(&[1, 2, 10, 20], Some(2), 40));
        assert!(!catalog.contains_next(&[1, 2, 10, 20], Some(2), 50));
        assert!(!catalog.contains_next(&[1, 2, 10, 99], Some(2), 30));
    }

    #[test]
    fn contains_next_can_fallback_to_first_root_token() {
        let mut catalog = Catalog::default();
        catalog.insert([10, 20, 30]);

        assert!(catalog.contains_next(&[1, 2, 10, 20], None, 30));
        assert!(!catalog.contains_next(&[1, 2, 10, 20], None, 40));
    }

    #[test]
    fn valid_next_ids_are_precomputed_in_sorted_order() {
        let mut catalog = Catalog::default();
        catalog.insert([10, 40]);
        catalog.insert([10, 20]);
        catalog.insert([10, 30]);
        catalog.finalize();

        assert_eq!(
            catalog.prefix(&[10], Some(0)).valid_next_sorted(),
            &[20, 30, 40],
        );
    }
}
