from __future__ import annotations

from cvstudio.core.op_docs import (
    FEATURE_TOPICS,
    OP_DOCS,
    documented_op_ids,
    feature_topic_keys,
    get_feature_doc,
    get_op_doc,
)
from cvstudio.core.pipeline import SOURCE_SPEC
from cvstudio.operations import all_builtin_specs


def test_every_registered_op_has_documentation() -> None:
    """Catch the easy regression of adding a new built-in op without writing
    its help entry."""
    expected_ids = {SOURCE_SPEC.id, *(spec.id for spec in all_builtin_specs())}
    missing = expected_ids - set(documented_op_ids())
    assert not missing, (
        f"These op ids have no entry in OP_DOCS: {sorted(missing)}"
    )


def test_get_op_doc_returns_html_for_known_id() -> None:
    blob = get_op_doc("filtering.gaussian_blur")
    assert "<h2>Gaussian Blur" in blob
    assert "Kernel size" in blob


def test_get_op_doc_falls_back_for_unknown_id() -> None:
    blob = get_op_doc("not.a.real.op")
    assert "No documentation" in blob
    assert "not.a.real.op" in blob


def test_documentation_entries_are_non_trivial() -> None:
    """Every entry should be more than just a heading — guards against
    placeholder strings sneaking in."""
    short = [
        spec_id
        for spec_id, body in OP_DOCS.items()
        if len(body.strip()) < 120
    ]
    assert not short, f"Doc entries that look too short: {short}"


def test_documentation_uses_qtextbrowser_friendly_tags_only() -> None:
    """Limits content to the small HTML subset QTextBrowser renders well —
    catches accidental Markdown or unsupported tags."""
    allowed = {"h2", "h3", "p", "ul", "ol", "li", "b", "i", "code", "br", "em", "strong"}
    import re

    tag_pattern = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)\b")
    for spec_id, body in OP_DOCS.items():
        tags = set(tag_pattern.findall(body))
        offending = tags - allowed
        assert not offending, (
            f"{spec_id} uses tags {sorted(offending)} outside the allowed subset"
        )
    for topic in FEATURE_TOPICS:
        tags = set(tag_pattern.findall(topic.body))
        offending = tags - allowed
        assert not offending, (
            f"feature {topic.key!r} uses tags {sorted(offending)} outside the allowed subset"
        )


def test_each_feature_topic_resolves_via_get_feature_doc() -> None:
    for key in feature_topic_keys():
        html = get_feature_doc(key)
        assert "<h2>" in html, f"feature {key} should start with an <h2> heading"


def test_app_covers_each_named_feature_area() -> None:
    """Catch the obvious regression of dropping a key topic — every major
    user-facing area should have its own help entry."""
    expected = {
        "getting_started",
        "opening_files",
        "pipeline_editor",
        "image_view",
        "histogram",
        "tools_panel",
        "roi",
        "save_image",
        "record_video",
        "save_load_pipeline",
        "export_code",
        "dataset_gallery",
        "batch",
    }
    missing = expected - set(feature_topic_keys())
    assert not missing, f"FEATURE_TOPICS missing keys: {sorted(missing)}"
