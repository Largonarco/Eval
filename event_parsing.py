def get_blocks(events: list[dict], exclude_hero_title: bool = True) -> list[dict]:
    blocks = [event for event in events if event["type"] == "block"]
    if exclude_hero_title:
        blocks = [
            block
            for block in blocks
            if block.get("content", {}).get("source", None)
            not in ("HeroImageTool", "TitleTool")
        ]
    return blocks


def extract_content(block: dict) -> dict:
    content = block["content"]
    output_type = content["output_type"]
    if output_type == "header":
        return {"header": content["output"]}
    elif output_type == "text":
        return {"paragraph": content["output"]}
    elif output_type == "data":
        return {"table": content["output"]}
    elif output_type == "metric":
        return {
            "number": content["output"]["number"],
            "description": content["output"]["description"],
        }
    elif output_type == "image" and content["output_subtype"] == "ai_generated_image":
        return {"ai_generated_image": content["llm_prompt"]}
    elif output_type == "image" and content["output_subtype"] == "image":
        return {"google_image": content["query"]}
    elif output_type == "image" and content["output_subtype"] == "chart_image":
        return {"chart": content["query"], "caption": content["caption"]}
    elif output_type == "quote":
        return {"quote": content["output"], "author": content["author"]}
    elif output_type == "tweet":
        # TODO
        return {}
    elif output_type == "title":
        return {"title": content["output"]}
    else:
        raise ValueError(f"Encountered block with invalid type: {block}")
