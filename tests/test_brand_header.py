from __future__ import annotations

import re
from html.parser import HTMLParser


class HeaderParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def test_brand_header_is_static_accessible_and_keeps_product_copy():
    import app

    ui = app.build_app()
    headers = [
        component["props"]["value"]
        for component in ui.get_config_file()["components"]
        if component["type"] == "html"
        and "id='muan-global-nav'" in component.get("props", {}).get("value", "")
    ]
    assert len(headers) == 1
    parser = HeaderParser()
    parser.feed(headers[0])
    assert parser.text == [
        "暮安智护", "MUAN CARE", "本地 AI 安全监测", "跌倒全周期风险监测",
        "前置风险评估 · 跌倒事件识别 · 本地隐私处理",
    ]
    assert any(tag == "svg" and attrs.get("aria-hidden") == "true"
               and attrs.get("focusable") == "false" for tag, attrs in parser.tags)
    assert not any(tag in {"script", "link", "img", "button", "a"}
                   for tag, _ in parser.tags)
    assert not any(key.startswith("on") or key in {"src", "href"}
                   for _, attrs in parser.tags for key in attrs)


def test_brand_foregrounds_are_explicit_and_high_contrast():
    from app import CSS

    def declarations(selector):
        return re.search(re.escape(selector) + r"\s*\{([^}]+)\}", CSS)[1]

    def luminance(color):
        channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                  for value in channels]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    background = re.search(r"background:\s*(#[0-9a-f]{6})", declarations("#muan-global-nav"))[1]
    for name in ("brand-name", "brand-english", "nav-meta"):
        rule = declarations(f"#muan-global-nav .{name}")
        foreground = re.search(r"color:\s*(#[0-9a-f]{6})\s*!important", rule)[1]
        contrast = (luminance(foreground) + 0.05) / (luminance(background) + 0.05)
        assert contrast >= 4.5
