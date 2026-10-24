"""Single policy registry shared by setup, voice discovery and cloning.

Unknown characters can speak Japanese with credit, but cannot be cloned.
The cloning decisions below follow the approved Aya voice contract.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Character:
    slug: str
    terms_url: str
    gender: str = "female"
    ja_allowed: bool = True
    clone_allowed: bool = True
    note: str = "Personal use with credit"
    credit: str = ""


CHARACTERS = {
    "四国めたん": Character("shikoku-metan", "https://zunko.jp/con_ongen_kiyaku.html"),
    "ずんだもん": Character("zundamon", "https://zunko.jp/con_ongen_kiyaku.html"),
    "春日部つむぎ": Character("kasukabe-tsumugi", "https://tsumugi-official.studio.site/rule"),
    "九州そら": Character("kyushu-sora", "https://zunko.jp/con_ongen_kiyaku.html"),
    "冥鳴ひまり": Character("meimei-himari", "https://meimeihimari.wixsite.com/himari/terms-of-use"),
    "波音リツ": Character("namine-ritsu", "https://www.canon-voice.com/kiyaku.html"),
    "春歌ナナ": Character("haruka-nana", "https://confetto.chu.jp/haruka/"),
    "小夜/SAYO": Character("sayo", "https://316soramegu.wixsite.com/sayo/guide"),
    "中国うさぎ": Character("chugoku-usagi", "https://zunko.jp/con_ongen_kiyaku.html"),
    "東北ずん子": Character("tohoku-zunko", "https://zunko.jp/con_ongen_kiyaku.html"),
    "東北きりたん": Character("tohoku-kiritan", "https://zunko.jp/con_ongen_kiyaku.html"),
    "東北イタコ": Character("tohoku-itako", "https://zunko.jp/con_ongen_kiyaku.html"),
    "櫻歌ミコ": Character("ouka-miko", "https://miko35.info/", note="Cloning explicitly allowed"),
    "雨晴はう": Character("amehare-hau", "https://amehau.com/", clone_allowed=False,
                          note="Voice changer / AI use excluded"),
    "猫使ビィ": Character("nekotsuka-bee", "https://nekonohako.fanbox.cc/posts/4624073",
                         clone_allowed=False, note="Voice changer / AI use excluded"),
    "猫使アル": Character("nekotsuka-aru", "https://nekonohako.fanbox.cc/posts/4624073",
                         clone_allowed=False, note="Voice changer / AI use excluded"),
    "もち子さん": Character("mochiko", "https://vtubermochio.wixsite.com/mochizora/利用規約",
                         clone_allowed=False, credit="VOICEVOX:もち子(cv 明日葉よもぎ)",
                         note="Voice changer / AI use excluded"),
    "栗田まろん": Character("kurita-maron", "https://aivoice.jp/character/maron/", gender="male",
                         clone_allowed=False, note="Clone terms unverified"),
    "琴詠ニア": Character("kotoyomi-nia", "https://commons.nicovideo.jp/works/nc315435",
                        clone_allowed=False, note="Clone terms unverified"),
    "玄野武宏": Character("kurono-takehiro", "https://virvoxproject.wixsite.com/official/",
                         gender="male", clone_allowed=False),
    "白上虎太郎": Character("shirakami-kotaro", "https://virvoxproject.wixsite.com/official/",
                         gender="male", clone_allowed=False),
    "青山龍星": Character("aoyama-ryusei", "https://virvoxproject.wixsite.com/official/",
                         gender="male", clone_allowed=False),
    "剣崎雌雄": Character("kenzaki-mesuo", "https://virvoxproject.wixsite.com/official/",
                         gender="male", clone_allowed=False),
}


def policy(name: str) -> Character:
    return CHARACTERS.get(name, Character("", "https://voicevox.hiroshiba.jp/",
                                         gender="unknown", clone_allowed=False,
                                         note="Unknown character: Japanese only"))


def clone_characters():
    return ((name, item) for name, item in CHARACTERS.items()
            if item.clone_allowed and item.gender == "female")


def voice_metadata(name: str) -> dict:
    item = policy(name)
    return {"gender": item.gender, "credit": item.credit or f"VOICEVOX:{name}",
            "terms_url": item.terms_url}
