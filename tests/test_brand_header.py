from __future__ import annotations

import base64
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


def test_welcome_brand_and_local_artwork_do_not_need_network():
    import app

    config = app.build_app().get_config_file()
    welcome = next(c for c in config['components'] if c.get('props', {}).get('elem_id') == 'welcome-block')
    parser = HeaderParser()
    parser.feed(welcome['props']['value'])
    text = ''.join(parser.text)
    assert '暮安智护' in text and '老年人跌倒检测平台' in text
    assert '让每一步，' in text and '都多一份安心。' in text
    assert any(attrs.get('class') == 'ma-moon' and attrs.get('aria-hidden') == 'true' for _, attrs in parser.tags)
    assert any(tag == 'button' and attrs.get('id') == 'muan-enter' and attrs.get('type') == 'button' for tag, attrs in parser.tags)
    images = [attrs for tag, attrs in parser.tags if tag == 'img']
    assert len(images) == 1 and images[0]['alt'].startswith('品牌插画')
    assert images[0]['src'].startswith('data:image/jpeg;base64,')
    image = base64.b64decode(images[0]['src'].split(',', 1)[1], validate=True)
    assert image == (app.ASSETS_DIR / 'muan-dusk-brand.jpg').read_bytes()
    assert not any(tag in {'script', 'link', 'iframe'} for tag, _ in parser.tags)
    assert not any(key.startswith('on') or value.startswith(('http:', 'https:'))
                   for _, attrs in parser.tags for key, value in attrs.items() if value)
    assert welcome['props']['js_on_load'] == app.NAVIGATION_JS


def test_brand_foregrounds_are_explicit_and_high_contrast():
    from app import CSS

    def luminance(color):
        channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                  for value in channels]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    for foreground, background in [('#23394e', '#fffdfa'), ('#5f6f7c', '#f8f5ef'), ('#f4f3ee', '#223b54'), ('#b7c8d7', '#223b54'), ('#203248', '#edc38f')]:
        assert foreground in CSS and background in CSS
        light, dark = sorted([luminance(foreground), luminance(background)], reverse=True)
        assert (light + 0.05) / (dark + 0.05) >= 4.5
    assert re.search(r'\.ma-moon\s*\{[^}]*radial-gradient', CSS)


def test_sidebar_keeps_native_tabs_and_history_select_callbacks():
    import app

    ui = app.build_app()
    config = ui.get_config_file()
    tabs = [c for c in config['components'] if c['type'] == 'tabitem']
    assert [(c['props']['label'], c['props']['id']) for c in tabs] == [
        ('视频检测', 'video'), ('实时监测', 'live'), ('事件中心', 'events'), ('前置风险记录', 'risk'),
    ]
    by_id = {c['props']['id']: c['id'] for c in tabs}
    callbacks = {fn.fn.__name__: fn for fn in ui.fns.values() if fn.fn}
    assert callbacks['open_event_history'].targets == [(by_id['events'], 'select')]
    assert callbacks['open_pre_fall_history'].targets == [(by_id['risk'], 'select')]
    assert not callbacks['open_event_history'].queue
    assert not callbacks['open_pre_fall_history'].queue
