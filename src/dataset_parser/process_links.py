import json
import logging
import time

import requests

from src.kg_builder.config import (
    DEFAULT_LINK_TYPES,
    LLM_BACKEND,
    LLM_CONCURRENCY,
    LLM_MODEL_PATH,
    LLM_N_BATCH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_TEMPERATURE,
)
from src.shared.llm_service import LLMService

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "WikiLinkClassifier/1.1 (mailto:your_email@example.com)"}


def normalize_title(title):
    """Преобразует название из формата файла/json в формат для API."""
    if not title:
        return ""
    return title.replace("_", " ").strip()


def get_qids_batch(titles, lang="ru"):
    url = "https://www.wikidata.org/w/api.php"
    title_to_qid = {}

    chunk_size = 50
    for i in range(0, len(titles), chunk_size):
        chunk = titles[i : i + chunk_size]
        logger.info(f"Запрос Q-ID для батча из {len(chunk)} статей...")
        titles_str = "|".join(chunk)

        params = {
            "action": "wbgetentities",
            "sites": f"{lang}wiki",
            "titles": titles_str,
            "languages": lang,
            "format": "json",
        }

        try:
            response = requests.get(url, params=params, headers=HEADERS).json()
            if "entities" in response:
                for qid, entity in response["entities"].items():
                    if qid != "-1":
                        title = (
                            entity.get("sitelinks", {})
                            .get(f"{lang}wiki", {})
                            .get("title")
                        )
                        if title:
                            title_to_qid[title] = qid
        except Exception as e:
            logger.error(f"Ошибка API: {e}")

        time.sleep(0.5)

    return title_to_qid


def get_bulk_relations(qid_pairs, lang="ru"):
    if not qid_pairs:
        return {}

    url = "https://query.wikidata.org/sparql"
    relations_map = {}
    chunk_size = 50  # Разбиваем на батчи, чтобы не получить ошибку слишком длинного URI

    for i in range(0, len(qid_pairs), chunk_size):
        chunk = qid_pairs[i : i + chunk_size]
        values_str = " ".join([f"(wd:{src} wd:{tgt})" for src, tgt in chunk])

        query = f"""
        SELECT ?source ?target ?propertyLabel WHERE {{
          VALUES (?source ?target) {{ {values_str} }}
          ?source ?wdt_prop ?target .
          ?property wikibase:directClaim ?wdt_prop .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
        }}
        """

        try:
            # Используем POST вместо GET
            post_headers = {
                "User-Agent": HEADERS["User-Agent"],
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }
            response = requests.post(
                url, data={"query": query, "format": "json"}, headers=post_headers
            )

            if response.status_code != 200:
                logger.error(
                    f"Ошибка SPARQL {response.status_code}: {response.text[:100]}"
                )
                continue

            results = response.json().get("results", {}).get("bindings", [])

            for item in results:
                src_id = item["source"]["value"].split("/")[-1]
                tgt_id = item["target"]["value"].split("/")[-1]
                relation = item["propertyLabel"]["value"]

                pair = (src_id, tgt_id)
                if pair not in relations_map:
                    relations_map[pair] = []
                relations_map[pair].append(relation)

        except Exception as e:
            logger.error(f"Ошибка SPARQL-запроса: {e}")

        time.sleep(1)

    return relations_map


def process_links_with_wikidata(
    input_file, output_file, lang="ru", use_llm_fallback=False
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Собираем нормализованные заголовки для API
    all_titles = set()
    for source, links in data.items():
        all_titles.add(normalize_title(source))
        for link in links:
            all_titles.add(normalize_title(link["target"]))

    logger.info(f"Найдено уникальных статей: {len(all_titles)}")

    # 2. Получаем Q-ID
    title_to_qid = get_qids_batch(list(all_titles), lang=lang)
    logger.info(f"Успешно получено Q-ID: {len(title_to_qid)}")

    # 3. Формируем пары Q-ID (ВАЖНО: нормализуем ключи при поиске в словаре)
    qid_pairs = set()
    for source, links in data.items():
        q_source = title_to_qid.get(normalize_title(source))
        if not q_source:
            continue

        for link in links:
            q_target = title_to_qid.get(normalize_title(link["target"]))
            if q_target:
                qid_pairs.add((q_source, q_target))

    # 4. Получаем связи
    logger.info(f"Отправляем SPARQL-запросы для {len(qid_pairs)} пар...")
    relations_map = get_bulk_relations(list(qid_pairs), lang=lang)

    # 5. Обновляем JSON (снова нормализуем ключи для поиска)
    count_classified = 0
    for source, links in data.items():
        q_source = title_to_qid.get(normalize_title(source))

        for link in links:
            q_target = title_to_qid.get(normalize_title(link["target"]))
            relation_type = None

            if q_source and q_target:
                pair = (q_source, q_target)
                if pair in relations_map:
                    relation_type = ", ".join(relations_map[pair])
                    count_classified += 1

            link["relation_type"] = relation_type
            link["q_source"] = q_source
            link["q_target"] = q_target

    # 6. LLM Fallback for unclassified links
    unclassified_links = []
    link_refs = []  # Keep references to update the original dict
    for source, links in data.items():
        for link in links:
            if not link.get("relation_type"):
                unclassified_links.append(
                    {
                        "source": source,
                        "target": link["target"],
                        "context": link.get("context", ""),
                    }
                )
                link_refs.append(link)
    logger.info(f"Найдено {len(unclassified_links)} неклассифицированных ссылок.")
    if unclassified_links and use_llm_fallback:
        logger.info("Запуск LLM Fallback...")

        llm_service = LLMService(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            n_ctx=LLM_N_CTX,
            n_batch=LLM_N_BATCH,
            temperature=LLM_TEMPERATURE,
            use_api=False,  # Configure via CLI args if necessary
            backend=LLM_BACKEND,
            concurrency=LLM_CONCURRENCY,
            default_link_types=DEFAULT_LINK_TYPES,
        )

        llm_results = llm_service.batch_classify_context_link(
            contexts=unclassified_links,
            relation_types=DEFAULT_LINK_TYPES,
        )

        llm_classified_count = 0
        for ref, result in zip(link_refs, llm_results):
            if result:
                ref["relation_type"] = result
                llm_classified_count += 1

        logger.info(
            f"LLM классифицировал {llm_classified_count} из {len(unclassified_links)} ссылок."
        )
        llm_service.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logger.info(f"Готово! Классифицировано {count_classified} ссылок через Wikidata.")
    logger.info(f"Результаты сохранены в {output_file}")
