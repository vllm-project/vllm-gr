if [ -z "$1" ]; then
  echo "Error: PR number argument is missing."
  echo "Usage: $0 <pr_number>"
  exit 1
fi

# curl -L \
#   -H "Accept: application/vnd.github+json" \
#   -H "Authorization: Bearer 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" \
#   -H "X-GitHub-Api-Version: 2022-11-28" \
#   "https://api.github.com/repos/JiusiServe/vllm-gr/pulls/$1/requested_reviewers"



curl -L \
  -X DELETE \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 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" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/JiusiServe/vllm-gr/pulls/$1/requested_reviewers" \
  -d '{"reviewers":["hsliuustc0106"]}'
