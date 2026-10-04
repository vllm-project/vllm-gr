// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#define MAX_DISTINCT_RUNS 512

typedef struct {
  Py_ssize_t start;
  Py_ssize_t len;
} TextSlice;

// Record each element in the output sequence:
// type == 0: integer token_id directly
// type == 1: reference to distinct text run index (0 .. num_distinct-1)
typedef struct {
  int32_t type;
  int32_t val;
} SequenceItem;

static PyObject* py_fast_sid_encode(PyObject* self, PyObject* args) {
  (void)self;
  const char* text;
  Py_ssize_t len;
  const char* lut_a_bytes;
  Py_ssize_t lut_a_len;
  const char* lut_b_bytes;
  Py_ssize_t lut_b_len;
  const char* lut_c_bytes;
  Py_ssize_t lut_c_len;
  int32_t space_id;
  int32_t sid_begin_id;
  int32_t sid_end_id;
  PyObject* added_dict;       // full dict of {str: token_id}
  PyObject* non_sid_dict;     // non-sid dict {str: token_id}
  PyObject* encode_batch_fn;  // tokenizer._tokenizer.encode_batch

  if (!PyArg_ParseTuple(args, "s#y#y#y#iiiOOO", &text, &len, &lut_a_bytes,
                        &lut_a_len, &lut_b_bytes, &lut_b_len, &lut_c_bytes,
                        &lut_c_len, &space_id, &sid_begin_id, &sid_end_id,
                        &added_dict, &non_sid_dict, &encode_batch_fn)) {
    return NULL;
  }

  if (len == 0) {
    Py_RETURN_NONE;
  }

  const int32_t* lut_a = (const int32_t*)lut_a_bytes;
  Py_ssize_t lut_a_count = lut_a_len / sizeof(int32_t);
  const int32_t* lut_b = (const int32_t*)lut_b_bytes;
  Py_ssize_t lut_b_count = lut_b_len / sizeof(int32_t);
  const int32_t* lut_c = (const int32_t*)lut_c_bytes;
  Py_ssize_t lut_c_count = lut_c_len / sizeof(int32_t);

  TextSlice distinct_slices[MAX_DISTINCT_RUNS + 1];
  int num_distinct = 0;

  // Estimate items: at most len items
  if ((size_t)len > SIZE_MAX / sizeof(SequenceItem) - 16) {
    return PyErr_NoMemory();
  }
  SequenceItem* seq_items =
      (SequenceItem*)malloc(sizeof(SequenceItem) * (len + 16));
  if (!seq_items) {
    return PyErr_NoMemory();
  }
  Py_ssize_t num_items = 0;

  Py_ssize_t i = 0;
  while (i < len) {
    // Space fast path: only emit standalone space_id if the space is strictly
    // between two added tokens (e.g. "> <") so it does not alter
    // word/punctuation tokenization of preceding text runs.
    if (text[i] == ' ' && space_id >= 0 && i > 0 && text[i - 1] == '>' &&
        i + 1 < len && text[i + 1] == '<') {
      seq_items[num_items].type = 0;
      seq_items[num_items].val = space_id;
      num_items++;
      i++;
      continue;
    }

    if (text[i] == '<') {
      // Check <|sid_begin|> and <|sid_end|> directly (frequent in recommender
      // prompts)
      if (i + 1 < len && text[i + 1] == '|') {
        if (sid_begin_id >= 0 && i + 12 < len &&
            memcmp(text + i, "<|sid_begin|>", 13) == 0) {
          seq_items[num_items].type = 0;
          seq_items[num_items].val = sid_begin_id;
          num_items++;
          i += 13;
          continue;
        }
        if (sid_end_id >= 0 && i + 10 < len &&
            memcmp(text + i, "<|sid_end|>", 11) == 0) {
          seq_items[num_items].type = 0;
          seq_items[num_items].val = sid_end_id;
          num_items++;
          i += 11;
          continue;
        }
      }

      // Check SID token: <s_a_123>
      if (i + 5 < len && text[i + 1] == 's' && text[i + 2] == '_') {
        char lvl = text[i + 3];
        if (text[i + 4] == '_' && (lvl == 'a' || lvl == 'b' || lvl == 'c')) {
          Py_ssize_t start_num = i + 5;
          int64_t num = 0;
          int overflow = 0;
          Py_ssize_t cur = start_num;
          while (cur < len && text[cur] >= '0' && text[cur] <= '9') {
            if (num > 10000000 || cur - start_num > 7) {
              overflow = 1;
            } else {
              num = num * 10 + (text[cur] - '0');
            }
            cur++;
          }
          if (cur - start_num > 1 && text[start_num] == '0') {
            overflow = 1;
          }
          if (!overflow && cur > start_num && cur < len && text[cur] == '>') {
            const int32_t* lut =
                (lvl == 'a') ? lut_a : ((lvl == 'b') ? lut_b : lut_c);
            Py_ssize_t lut_count =
                (lvl == 'a') ? lut_a_count
                             : ((lvl == 'b') ? lut_b_count : lut_c_count);
            if (num < lut_count) {
              int32_t tid = lut[num];
              if (tid >= 0) {
                seq_items[num_items].type = 0;
                seq_items[num_items].val = tid;
                num_items++;
                i = cur + 1;
                continue;
              }
            }
            // Non-contiguous or out-of-LUT: lookup in added_dict
            PyObject* key = PyUnicode_FromStringAndSize(text + i, cur - i + 1);
            if (!key) {
              free(seq_items);
              return NULL;
            }
            PyObject* tid = PyDict_GetItem(added_dict, key);
            Py_DECREF(key);
            if (tid) {
              seq_items[num_items].type = 0;
              seq_items[num_items].val = (int32_t)PyLong_AsLong(tid);
              num_items++;
              i = cur + 1;
              continue;
            }
          }
        }
      }

      // Check non-sid added tokens via non_sid_dict (e.g. ChatML <|im_start|>)
      Py_ssize_t close_bracket = -1;
      for (Py_ssize_t j = i + 1; j < len && j < i + 64; j++) {
        if (text[j] == '>') {
          close_bracket = j;
          break;
        }
      }
      if (close_bracket != -1) {
        PyObject* key =
            PyUnicode_FromStringAndSize(text + i, close_bracket - i + 1);
        if (!key) {
          free(seq_items);
          return NULL;
        }
        PyObject* tid = PyDict_GetItem(non_sid_dict, key);
        Py_DECREF(key);
        if (tid) {
          seq_items[num_items].type = 0;
          seq_items[num_items].val = (int32_t)PyLong_AsLong(tid);
          num_items++;
          i = close_bracket + 1;
          continue;
        }
      }
      // Stray '<' or unknown tag: abort and fallback
      free(seq_items);
      Py_RETURN_NONE;
    }

    // Plain text run: scan until next '<'
    Py_ssize_t run_start = i;
    while (i < len && text[i] != '<') {
      i++;
    }
    Py_ssize_t run_len = i - run_start;

    // Check if run already in distinct_slices
    int run_idx = -1;
    for (int d = 0; d < num_distinct; d++) {
      if (distinct_slices[d].len == run_len &&
          memcmp(text + distinct_slices[d].start, text + run_start, run_len) ==
              0) {
        run_idx = d;
        break;
      }
    }
    if (run_idx == -1) {
      if (num_distinct >= MAX_DISTINCT_RUNS) {
        // Too many distinct runs: abort and fallback
        free(seq_items);
        Py_RETURN_NONE;
      }
      run_idx = num_distinct;
      distinct_slices[num_distinct].start = run_start;
      distinct_slices[num_distinct].len = run_len;
      num_distinct++;
    }

    seq_items[num_items].type = 1;
    seq_items[num_items].val = run_idx;
    num_items++;
  }

  // Now batch-encode distinct text runs (if any)
  PyObject** distinct_id_lists = NULL;
  if (num_distinct > 0) {
    distinct_id_lists = (PyObject**)malloc(sizeof(PyObject*) * num_distinct);
    if (!distinct_id_lists) {
      free(seq_items);
      return PyErr_NoMemory();
    }

    PyObject* batch_arg = PyList_New(num_distinct);
    if (!batch_arg) {
      free(distinct_id_lists);
      free(seq_items);
      return NULL;
    }
    for (int d = 0; d < num_distinct; d++) {
      PyObject* s = PyUnicode_FromStringAndSize(text + distinct_slices[d].start,
                                                distinct_slices[d].len);
      if (!s) {
        Py_DECREF(batch_arg);
        free(distinct_id_lists);
        free(seq_items);
        return NULL;
      }
      PyList_SET_ITEM(batch_arg, d, s);  // steals ref
    }

    PyObject* call_args = PyTuple_Pack(1, batch_arg);
    Py_DECREF(batch_arg);
    if (!call_args) {
      free(distinct_id_lists);
      free(seq_items);
      return NULL;
    }
    PyObject* call_kwargs =
        Py_BuildValue("{s:O}", "add_special_tokens", Py_False);
    if (!call_kwargs) {
      Py_DECREF(call_args);
      free(distinct_id_lists);
      free(seq_items);
      return NULL;
    }
    PyObject* enc_res = PyObject_Call(encode_batch_fn, call_args, call_kwargs);
    Py_DECREF(call_args);
    Py_DECREF(call_kwargs);
    if (!enc_res) {
      free(distinct_id_lists);
      free(seq_items);
      return NULL;
    }
    if (!PyList_Check(enc_res) || PyList_Size(enc_res) != num_distinct) {
      PyErr_SetString(
          PyExc_TypeError,
          "encode_batch must return a list of length equal to distinct inputs");
      Py_DECREF(enc_res);
      free(distinct_id_lists);
      free(seq_items);
      return NULL;
    }

    for (int d = 0; d < num_distinct; d++) {
      PyObject* enc = PyList_GetItem(enc_res, d);
      if (!enc) {
        for (int k = 0; k < d; k++) Py_DECREF(distinct_id_lists[k]);
        Py_DECREF(enc_res);
        free(distinct_id_lists);
        free(seq_items);
        return NULL;
      }
      PyObject* ids = PyObject_GetAttrString(enc, "ids");
      if (!ids || !PyList_Check(ids)) {
        Py_XDECREF(ids);
        for (int k = 0; k < d; k++) Py_DECREF(distinct_id_lists[k]);
        Py_DECREF(enc_res);
        free(distinct_id_lists);
        free(seq_items);
        if (!PyErr_Occurred()) {
          PyErr_SetString(PyExc_TypeError, "Encoding.ids must be a list");
        }
        return NULL;
      }
      distinct_id_lists[d] = ids;
    }
    Py_DECREF(enc_res);
  }

  // Precompute total output length and assemble final PyList
  Py_ssize_t total_tokens = 0;
  for (Py_ssize_t idx = 0; idx < num_items; idx++) {
    if (seq_items[idx].type == 0) {
      total_tokens++;
    } else {
      total_tokens += PyList_Size(distinct_id_lists[seq_items[idx].val]);
    }
  }

  PyObject* out_list = PyList_New(total_tokens);
  if (!out_list) {
    if (distinct_id_lists) {
      for (int d = 0; d < num_distinct; d++) Py_DECREF(distinct_id_lists[d]);
      free(distinct_id_lists);
    }
    free(seq_items);
    return NULL;
  }

  Py_ssize_t out_idx = 0;
  for (Py_ssize_t item_idx = 0; item_idx < num_items; item_idx++) {
    if (seq_items[item_idx].type == 0) {
      PyObject* v = PyLong_FromLong(seq_items[item_idx].val);
      if (!v) {
        goto assemble_error;
      }
      PyList_SET_ITEM(out_list, out_idx++, v);
    } else {
      int d = seq_items[item_idx].val;
      PyObject* id_list = distinct_id_lists[d];
      Py_ssize_t sub_len = PyList_Size(id_list);
      for (Py_ssize_t s = 0; s < sub_len; s++) {
        PyObject* token_val = PyList_GetItem(id_list, s);
        Py_INCREF(token_val);
        PyList_SET_ITEM(out_list, out_idx++, token_val);
      }
    }
  }

  if (distinct_id_lists) {
    for (int d = 0; d < num_distinct; d++) Py_DECREF(distinct_id_lists[d]);
    free(distinct_id_lists);
  }
  free(seq_items);
  return out_list;

assemble_error:
  if (distinct_id_lists) {
    for (int d = 0; d < num_distinct; d++) Py_DECREF(distinct_id_lists[d]);
    free(distinct_id_lists);
  }
  free(seq_items);
  Py_DECREF(out_list);
  return NULL;
}

static PyMethodDef FastSidMethods[] = {
    {"fast_sid_encode", py_fast_sid_encode, METH_VARARGS,
     "Fast SID encode in C using direct LUT and item markers"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef fastsidmodule = {
    PyModuleDef_HEAD_INIT, .m_name = "_fast_sid_c",     .m_doc = NULL,
    .m_size = -1,          .m_methods = FastSidMethods, .m_slots = NULL,
};

PyMODINIT_FUNC PyInit__fast_sid_c(void) {
  return PyModule_Create(&fastsidmodule);
}
