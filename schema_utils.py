def clean_openapi_schema(schema):
    """
    Recursively remove OpenAPI/JSON Schema fields that Gemini function calling API does not support.
    Removes: $schema, additional_properties, and any unknown fields.
    """
    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    for k, v in schema.items():
        if k in {"$schema", "additional_properties"}:
            continue
        # OpenAPI/JSON Schema uses 'additionalProperties', but Gemini expects only 'properties' and 'required'
        if k.lower() == "additionalproperties":
            continue
        if isinstance(v, dict):
            cleaned[k] = clean_openapi_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [clean_openapi_schema(i) for i in v]
        else:
            cleaned[k] = v
    return cleaned
