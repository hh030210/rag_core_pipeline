# RAG 接口说明
---

## 1. 检索

`POST /retrieve`

```json
{
  "query": "大成殿是什么",
  "top_k": 5,
  "collection": "travel_agent_1001",
  "where": {"source_file": "景点介绍.txt"}
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `query` | 是 | 检索文本 |
| `top_k` | 是 | 返回条数 |
| `collection` | 是 | 集合名 |
| `where` | 否 | 元数据等值过滤。省略则搜整个集合。值是字符串、数字或布尔 |

```json
{
  "results": [
    {
      "query_text": "大成殿是什么",
      "hits": [
        {
          "id": "vec_a1b2c3d4e5f67890",
          "text": "大成殿是家庙主殿，取义孟子“孔子之谓集大成”。殿内高悬康熙御笔“万世师表”匾额，乾隆题联：“气备四时与天地鬼神日月合其德，教垂万世继尧舜禹汤文武作之师。",
          "score": 0.82,
          "metadata": {
            "score_type": "doc",
            "filename": "景点介绍.txt",
            "source_file": "景点介绍.txt",
            "content_tags": "poi_intro,internal_layout",
            "poi_name": "大成殿"
          }
        }
      ]
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `hits[].id` | 该条向量的 id |
| `hits[].text` | 正文 |
| `hits[].score` | 相似度，越大越相近。本接口按 `top_k` 返回 |
| `hits[].metadata.score_type` | `doc` |
| `hits[].metadata.filename` | 源文件名 |
| `hits[].metadata.source_file` | 来源文件名，过滤和删除用 |
| `hits[].metadata.content_tags` | 标签，逗号分隔 |
| `hits[].metadata.poi_name` | 景点名，可空 |

---

## 2. 写入

`POST /collections/{collection}/documents`

`documents`、`metadatas`、`ids` 三个数组等长。`ids` 可以不传。metadata 的值是字符串、数字或布尔。单批建议不超过 16 条。成功时 `status` 为 `success`。

```json
{
  "documents": ["景区：衢州古城文化旅游区\n分类：大成殿"],
  "metadatas": [{
    "score_type": "doc",
    "filename": "景点介绍.txt",
    "source_file": "景点介绍.txt",
    "content_tags": "poi_intro,internal_layout",
    "poi_name": "大成殿"
  }],
  "ids": ["vec_a1b2c3d4e5f67890"]
}
```

| 字段 | 说明 |
|------|------|
| `documents[i]` | 正文，检索命中的 `text` 即此字段 |
| `metadatas[i].score_type` | `doc` |
| `metadatas[i].filename` | 源文件名 |
| `metadatas[i].source_file` | 来源文件名 |
| `metadatas[i].content_tags` | 标签，逗号分隔 |
| `metadatas[i].poi_name` | 景点名，可空 |
| `ids[i]` | 该条向量的 id，与检索结果 `hits[].id` 相同 |

---

## 3. 查询、更新、删除

`GET /collections/{collection}/documents`

查询参数：`limit`，以及 `where`（JSON 字符串）。

`PUT /collections/{collection}/documents`

```json
{
  "where": {"source_file": "景点介绍.txt"},
  "metadata_patch": {"content_tags": "poi_intro,internal_layout"}
}
```

`DELETE /collections/{collection}/documents`

```json
{"where": {"source_file": "景点介绍.txt"}}
```

`where` 为元数据等值条件，例如 `source_file`、`filename`。

---

## 4. 集合与就绪

`GET /ready`

就绪时 `status` 为 `success`。

`GET /collections`

返回集合名列表。

`POST /collections`

```json
{"collection_name": "travel_agent_1001", "metadata": {"score_type": "doc"}}
```

集合已存在时按成功处理。

`DELETE /collections/{collection_name}`

删除该集合及其全部向量。

`POST /embed`

```json
{"inputs": ["大成殿是什么", "门票多少钱"]}
```

```json
{"embeddings": [[0.01, 0.02], [0.03, 0.04]]}
```
