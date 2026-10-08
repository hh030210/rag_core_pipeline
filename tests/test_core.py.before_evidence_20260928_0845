import json
import tempfile
import unittest
from pathlib import Path

from rag_core.cluster_prompting import ClusterPromptPipeline, ClusterPromptRouter
from rag_core.dimension_stage import default_schema, build_inverted_index, build_tags
from rag_core.embedding import EmbeddingModel
from rag_core.evaluation import evaluate_row, load_gold_records, map_gold_to_chunks, route_metrics
from rag_core.prompting import PromptOptimizer
from rag_core.prompt_module import PromptOptimizationModule
from rag_core.retrieval import Retriever
from rag_core.settings import Settings
from rag_core.storage import make_payload
from rag_core.schema_v2 import load_schema
from rag_core.llm import LLMClient
from rag_core.integrated_chunker import OpenAICompatClient


class CorePipelineTests(unittest.TestCase):
    def test_v4_strict_intents_do_not_turn_subject_nouns_into_dimensions(self):
        from code_jyx.query_parser_v4 import QueryParserV4

        self.assertEqual(
            set(QueryParserV4._strict_intents("这座陵墓修建了多久？")),
            {"historical_event"},
        )
        self.assertEqual(
            set(QueryParserV4._strict_intents("学生票有优惠吗？")),
            {"ticket_policy"},
        )
        self.assertNotIn(
            "entity_type",
            QueryParserV4._strict_intents("这座建筑有哪些艺术特色？"),
        )

    def test_retriever_runs_v4_deterministic_query_recovery_without_llm(self):
        def node(node_id, name, parent_id, level, indexable, description):
            return {
                "id": node_id,
                "name": name,
                "parent_id": parent_id,
                "level": level,
                "indexable": indexable,
                "description": description,
                "aliases": [],
                "value_policy": "multi",
                "max_labels": 5,
            }

        schema = {
            "schema_version": "2.0",
            "dimensions": [
                node("operation_service", "开放运营", None, 1, False, "开放与游客服务"),
                node("opening_period", "开放时段", "operation_service", 2, True, "每日开放时间"),
                node("visitor_service", "游客服务", "operation_service", 2, True, "游客设施与服务规则"),
                node("ticketing_policy", "票务规则", None, 1, False, "价格和购票政策"),
                node("ticket_type", "票价类型", "ticketing_policy", 2, True, "门票价格类别"),
                node("ticket_policy", "购票政策", "ticketing_policy", 2, True, "预约和购票规定"),
                node("geographic_location", "地理位置", None, 1, False, "位置与交通"),
                node("transportation_mode", "交通方式", "geographic_location", 2, True, "到达交通手段"),
                node("history_culture", "历史文化", None, 1, False, "历史文化背景"),
                node("historical_event", "历史事件", "history_culture", 2, True, "兴建重修等历史事件"),
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "V_core_v2.json").write_text(
                json.dumps(schema, ensure_ascii=False), encoding="utf-8"
            )
            (root / "inverted_index_v2.json").write_text(
                json.dumps({
                    "opening_period": {}, "visitor_service": {},
                    "ticket_type": {}, "ticket_policy": {},
                    "transportation_mode": {}, "historical_event": {},
                }),
                encoding="utf-8",
            )
            (root / "local_points.json").write_text("[]", encoding="utf-8")
            settings = Settings(
                run_dir=root, backend="local", mock=True, vector_dim=8,
            )
            retriever = Retriever(
                run_dir=root,
                settings=settings,
                embeddings=EmbeddingModel(dimension=8, mock=True),
                llm=None,
            )

            self.assertIsNotNone(retriever.query_parser)
            self.assertIn("opening_period", retriever._parse_query("景区几点开门？")["constraints"])
            self.assertIn("ticket_policy", retriever._parse_query("怎么买票，需要预约吗？")["constraints"])
            self.assertIn("transportation_mode", retriever._parse_query("从火车站坐公交怎么去？")["constraints"])
            self.assertIn("visitor_service", retriever._parse_query("景区电话是多少？")["constraints"])
            self.assertIn("historical_event", retriever._parse_query("这座建筑修建了多久？")["constraints"])

    def test_dimension_policy_uses_main_dimension_for_reranking_not_filtering(self):
        policy = Retriever._dimension_policy({
            "constraints": {
                "location_hierarchy": {"labels": ["龙门石窟"], "role": "auxiliary"},
                "ticket_policy": {"intent_terms": ["预约"]},
                "visitor_service": {"intent_terms": ["电话"], "role": "main"},
            }
        })
        self.assertEqual(policy["mode"], "wide_recall_main_dimension_rerank")
        self.assertEqual(policy["main_dimension"], "visitor_service")
        self.assertEqual(policy["required_minimum"], 0)
        self.assertTrue(Retriever._passes_dimension_policy(
            policy, {"location_hierarchy"}
        ))

        three_intents = Retriever._dimension_policy({
            "constraints": {
                "ticket_policy": {"intent_terms": ["预约"], "role": "main"},
                "opening_period": {"intent_terms": ["开放"], "role": "secondary"},
                "visitor_service": {"intent_terms": ["电话"], "role": "secondary"},
            }
        })
        self.assertEqual(three_intents["main_dimension"], "ticket_policy")
        self.assertEqual(three_intents["secondary_dimensions"], ["opening_period", "visitor_service"])
        self.assertTrue(Retriever._passes_dimension_policy(
            three_intents, set()
        ))

    def test_canonical_label_resolver_matches_synonyms_within_dimension(self):
        from rag_core.dimension_labels import CanonicalLabelResolver

        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = CanonicalLabelResolver(temp_dir, {
                "person_role": ["帝王"],
                "ticket_policy": ["需要提前预约购票"],
            })
            self.assertEqual(
                resolver.canonical("person_role", "皇帝"),
                resolver.canonical("person_role", "帝王"),
            )
            self.assertEqual(
                resolver.canonical("ticket_policy", "预约"),
                resolver.canonical("ticket_policy", "需要提前预约购票"),
            )
            self.assertNotEqual(
                resolver.canonical("person_role", "皇帝"),
                resolver.canonical("entity_type", "皇帝"),
            )
            self.assertIn(
                "优惠",
                resolver.labels_in_text("ticket_policy", "大学生门票有优惠吗？"),
            )

    def test_typed_tag_vector_thresholds_are_dimension_specific(self):
        retriever = object.__new__(Retriever)
        retriever.settings = type("SettingsStub", (), {
            "tag_vector_threshold_profile": "typed_moderate"
        })()
        self.assertEqual(retriever._tag_vector_threshold("historical_event"), 0.54)
        self.assertEqual(retriever._tag_vector_threshold("ticket_policy"), 0.70)
        self.assertEqual(retriever._tag_vector_threshold("unknown_dimension"), 0.58)

    def test_cached_tag_vector_dot_preserves_cosine(self):
        from rag_core.retrieval import _dot_unit_vectors, _unit_vector

        left, right = [3.0, 4.0], [4.0, -3.0]
        self.assertAlmostEqual(_dot_unit_vectors(_unit_vector(left), _unit_vector(right)), 0.0)

    def test_poi_registry_resolves_aliases_to_parent_scenic_areas(self):
        from rag_core.poi_registry import PoiRegistry

        registry = PoiRegistry(["明十三陵", "颐和园", "西湖", "龙门石窟"])
        self.assertEqual(registry.scenic_scopes("长陵和定陵分别是哪位皇帝的陵寝？"), ["明十三陵"])
        self.assertEqual(registry.scenic_scopes("卢舍那大佛有多高？"), ["龙门石窟"])
        # A comparison question must retain both scopes instead of allowing
        # the later scenic-area mention to filter out the primary subject.
        self.assertEqual(
            registry.scenic_scopes("昆明湖的西堤为什么模仿西湖的苏堤？"),
            ["颐和园", "西湖"],
        )

    def test_fact_anchor_bonus_prefers_rare_named_fact_in_chunk(self):
        retriever = object.__new__(Retriever)
        retriever._anchor_df_cache = {}
        retriever.points = [
            {"payload": {"doc_title": "明十三陵", "chunk_text_full": "崇祯皇帝葬于思陵"}},
            {"payload": {"doc_title": "明十三陵", "chunk_text_full": "长陵是重要陵寝"}},
        ]
        terms = retriever._fact_anchor_terms("崇祯皇帝的陵墓规模怎么样？", {})
        self.assertIn("崇祯", terms)
        hit_bonus, hit_matches = retriever._fact_anchor_bonus(retriever.points[0]["payload"], terms)
        miss_bonus, _ = retriever._fact_anchor_bonus(retriever.points[1]["payload"], terms)
        self.assertGreater(hit_bonus, miss_bonus)
        self.assertTrue(any(item["term"] == "崇祯" for item in hit_matches))

    def test_chunk_recommendation_recovers_non_strict_proxy_json(self):
        response = (
            "推荐如下：\n```json\n"
            '{"genre":"技术文档","l_min":172,"l_max":860,'
            '"reasoning":"l_min = 0.6 \\* 286"}\n'
            "```\n补充说明"
        )
        result = OpenAICompatClient.parse_json_object(response)
        self.assertEqual(result["genre"], "技术文档")
        self.assertEqual(result["l_min"], 172)
        self.assertEqual(result["l_max"], 860)

    def test_chunk_recommendation_recovers_when_reasoning_quotes_break_json(self):
        response = (
            '{"genre":"技术文档","l_min":400,"l_max":800,'
            '"reasoning":"如"大成殿名称来源"等"}'
        )
        result = OpenAICompatClient.parse_json_object(response)
        self.assertEqual(result["l_min"], 400)
        self.assertEqual(result["l_max"], 800)

    def test_chunk_recommendation_normalizes_proxy_nested_shape(self):
        response = (
            '{"recommendation":{"genre":"技术文档",'
            '"length_range":[{"min":400,"max":900}],'
            '"reasoning":"简短理由"}}'
        )
        result = OpenAICompatClient.parse_json_object(response)
        self.assertEqual(result["genre"], "技术文档")
        self.assertEqual(result["l_min"], 400)
        self.assertEqual(result["l_max"], 900)

    def test_schema_is_two_level_and_valid(self):
        schema = load_schema(default_schema(), allow_legacy=False, strict=True)
        self.assertTrue(all(node["level"] in (1, 2) for node in schema["dimensions"]))
        self.assertTrue(all(node["indexable"] is (node["level"] == 2) for node in schema["dimensions"]))

    def test_multilabel_tags_keep_arrays(self):
        schema = default_schema()
        records = [{"doc_id": "d1", "doc_text": "儿童和青少年可在北京景点参观，上午开放，门票免费。"}]
        settings = Settings(run_dir=Path("/tmp/rag_core_test"), backend="local", mock=True)
        output = build_tags(records, schema, settings)
        tags = output["documents"]["d1"]["tags"]
        self.assertIn("儿童", tags["content.entity"])
        self.assertIn("青少年", tags["content.entity"])
        self.assertIsInstance(tags["content.entity"], list)
        index = build_inverted_index(output, schema)
        self.assertIn("儿童", index["postings"]["content.entity"])
        payload = make_payload({"chunk_id": "d1", "doc_id": "d1", "doc_text": records[0]["doc_text"]},
                               output["documents"]["d1"], schema)
        self.assertIsInstance(payload["dim_content.entity"], list)

    def test_mock_embedding_is_deterministic(self):
        model = EmbeddingModel(dimension=8, mock=True)
        self.assertEqual(model.encode(["a"])[0], model.encode(["a"])[0])

    def test_prompt_expansion_falls_back_without_llm(self):
        manager = PromptOptimizer(client=LLMClient(mock=True), run_dir=Path("/tmp/rag_core_test"))
        result = manager.expand("测试问题")
        self.assertEqual(result["sub_queries"], ["测试问题"])
        self.assertTrue(manager.load_module()["system_prompt"])

    def test_prompt_module_has_explicit_json_input_and_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "prompt_input.json"
            output_path = root / "prompt_output.json"
            input_path.write_text(json.dumps({
                "examples": [{
                    "question": "景区几点开放？",
                    "context": "景区每日八点开放。",
                    "reference_answer": "景区每日八点开放。",
                }],
                "config": {"iterations": 2},
            }, ensure_ascii=False), encoding="utf-8")
            module = PromptOptimizationModule(
                client=LLMClient(mock=True),
                work_dir=root / "work",
            )
            result = module.run_file(input_path, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["module_version"], "1.0")
            self.assertEqual(result["io"]["example_count"], 1)
            self.assertEqual(result["io"]["iterations"], 2)
            self.assertEqual(saved["method"], "no_examples_or_mock")
            self.assertTrue(saved["system_prompt"])

    def test_cluster_prompt_pipeline_and_router_work_in_mock_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            examples = [
                {"id": "q1", "question": "景区几点开放？", "reference_answer": "每日八点开放。",
                 "context": "景区每日八点开放。"},
                {"id": "q2", "question": "景区门票多少钱？", "reference_answer": "门票免费。",
                 "context": "景区门票免费。"},
                {"id": "q3", "question": "景区在哪里？", "reference_answer": "景区位于杭州。",
                 "context": "景区位于杭州。"},
                {"id": "q4", "question": "景区怎么去？", "reference_answer": "可乘公交到达。",
                 "context": "可乘公交到达景区。"},
            ]
            embeddings = EmbeddingModel(dimension=8, mock=True)
            pipeline = ClusterPromptPipeline(
                client=LLMClient(mock=True), embeddings=embeddings, run_dir=root,
            )
            result = pipeline.run(examples, cluster_count=2, iterations=1)
            self.assertEqual(result["metrics"]["sample_count"], 4)
            self.assertEqual(result["metrics"]["cluster_count_actual"], 2)
            self.assertTrue((root / "cluster_prompting.json").exists())
            self.assertEqual(len(result["question_cluster_mapping"]), 4)

            router = ClusterPromptRouter(
                run_dir=root, embeddings=embeddings, enabled=True, top_k=2,
            )
            routes = router.route("景区什么时候开放？")
            self.assertEqual(len(routes), 2)
            self.assertGreaterEqual(routes[0]["score"], routes[1]["score"])
            self.assertTrue(routes[0]["prompt"]["system_prompt"])

    def test_route_metrics_report_first_gold_and_cutoffs(self):
        result = route_metrics(
            [{"chunk_id": "noise"}, {"chunk_id": "gold"}, {"chunk_id": "other"}],
            ["gold"],
            ks=(1, 3),
        )
        self.assertEqual(result["first_gold_rank"], 2)
        self.assertEqual(result["rr"], 0.5)
        self.assertFalse(result["hit_at_1"])
        self.assertTrue(result["hit_at_3"])
        self.assertEqual(result["rr_at_1"], 0.0)
        self.assertEqual(result["rr_at_3"], 0.5)

    def test_gold_aliases_and_evidence_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gold.json"
            path.write_text(json.dumps([{
                "id": "q1",
                "query": "几点开放？",
                "source": ["景区每日八点开放。"],
                "attraction": "测试景区",
            }], ensure_ascii=False), encoding="utf-8")
            record = load_gold_records(path)[0]
            self.assertEqual(record["question"], "几点开放？")
            self.assertEqual(record["spot"], "测试景区")
            mapped = map_gold_to_chunks(record, [{
                "chunk_id": "c1",
                "source_file": "测试景区",
                "chunk_text_full": "介绍：景区每日八点开放。",
            }])
            self.assertEqual(mapped["gold_chunk_ids"], ["c1"])
            self.assertEqual(mapped["gold_resolution"]["match_basis"], "evidence_text")

    def test_evaluation_keeps_fusion_metrics_and_diagnostics_separate(self):
        row = evaluate_row(
            {"id": "q1", "question": "测试", "gold_chunk_ids": ["gold"]},
            {
                "semantic_candidates": [{"chunk_id": "noise"}, {"chunk_id": "gold"}],
                "dimension_candidates": [{"chunk_id": "gold"}],
                "fusion_candidates": [{"chunk_id": "noise"}, {"chunk_id": "gold"}],
            },
            ks=(1, 2),
            eval_depth=2,
        )
        self.assertEqual(row["retrieval_metrics"]["fusion"]["first_gold_rank"], 2)
        self.assertIn("fusion_diagnostics", row["retrieval_metrics"])
        self.assertEqual(row["retrieval_metrics"]["fusion_diagnostics"]["rank_gain"], -1)


if __name__ == "__main__":
    unittest.main()
