#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal /recommend client using requests."""

import json
import pathlib

import requests


api_url = "http://127.0.0.1:8000/recommend"
request_file = pathlib.Path(__file__).resolve().with_name("recommend_request.json")

payload = json.loads(request_file.read_text(encoding="utf-8"))
response = requests.post(api_url, json=payload, timeout=30)
response.raise_for_status()

data = response.json()
print(
    json.dumps(
        {
            "id": data.get("id"),
            "model": data.get("model"),
            "user_id": data.get("user_id"),
        },
        ensure_ascii=False,
    )
)

for idx, item in enumerate(data.get("recommendations", []), start=1):
    print(
        f"{idx}. item_id={item.get('item_id')} "
        f"score={item.get('score')} "
        f"index={item.get('index')}"
    )
