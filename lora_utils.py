"""LoRA trigger word categorization utilities for danbooru-style tags"""
import re

# --- General / non-character-specific tags ---

GENERAL_EXACT = {
    '1girl', '2girls', '3girls', '4girls', '5girls', '6+girls', 'multiple girls',
    '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys', 'multiple boys',
    '1other', 'androgynous', 'solo', 'solo focus', 'duo', 'group', 'couple',
    'full body', 'upper body', 'lower body', 'cowboy shot', 'close-up', 'portrait',
    'head shot', 'bust shot',
    'looking at viewer', 'looking away', 'looking back', 'looking down',
    'looking up', 'looking to the side', 'looking forward', 'looking afar',
    'looking off to the side', 'looking to the left', 'looking to the right',
    'smile', 'grin', 'smirk', 'open mouth', 'closed mouth', 'parted lips',
    'laughing', 'crying', 'tears', 'expressionless', 'serious', 'angry',
    'surprised', 'happy', 'sad', 'embarrassed', 'pout', 'blush',
    'nervous', 'flustered', 'determined', 'sleepy', 'drunk', 'dazed',
    'standing', 'sitting', 'lying', 'walking', 'running', 'kneeling',
    'crouching', 'jumping', 'floating', 'leaning forward', 'bending forward',
    'arms up', 'arm up', 'hands up', 'hand up', 'arms at sides',
    'hand on own hip', 'hand on hip', 'crossed arms', 'arms crossed',
    'waving', 'pointing', 'reaching out', 'holding',
    'outstretched arm', 'outstretched hand', 'spread arms',
    'v', 'peace sign', 'thumbs up', 'ok sign', 'finger gun',
    'outdoors', 'indoors', 'outside', 'inside',
    'simple background', 'white background', 'black background', 'grey background',
    'gray background', 'gradient background', 'blurry background', 'blurry',
    'depth of field',
    'day', 'night', 'dusk', 'dawn', 'sunrise', 'sunset',
    'sky', 'clouds', 'blue sky', 'cloudy sky', 'starry sky', 'scenery',
    'nature', 'city', 'cityscape', 'room', 'bedroom', 'classroom', 'office',
    'park', 'beach', 'forest', 'mountain', 'grass', 'flowers',
    'highres', 'absurdres', 'masterpiece', 'best quality', 'ultra-detailed',
    'detailed', '4k', '8k', 'hd',
    'monochrome', 'greyscale', 'grayscale',
    'chibi', 'deformed',
    'sweat', 'heart', 'sparkle',
    'censored', 'bar censor', 'mosaic censoring',
    'speech bubble', 'thought bubble',
    'comic', 'manga',
    'chibi', 'nendoroid',
}

# Substring matches for general tags
GENERAL_PARTIAL = [
    'background',
    'looking at',
    'looking to',
    'looking off',
    'from above',
    'from below',
    'from side',
    'from behind',
    'hand on',
    'arms at',
    'pov',
]

# --- Hair & Head ---

HAIR_HEAD_WORDS = {
    'hair', 'eye', 'eyes', 'eyebrow', 'eyebrows', 'eyelash', 'eyelashes',
    'eyelid', 'eyelids', 'pupil', 'pupils', 'iris',
    'forehead', 'bangs', 'ahoge', 'topknot',
    'hat', 'cap', 'headband', 'hairband', 'headdress', 'headwear',
    'crown', 'tiara', 'veil', 'hood',
    'ear', 'ears',
    'cheek', 'cheeks', 'lip', 'lips', 'nose', 'mouth', 'chin', 'jaw',
    'freckle', 'freckles', 'mole', 'sideburn', 'sideburns',
    'glasses', 'sunglasses', 'monocle', 'eyepatch',
    'horn', 'horns', 'halo',
    'mask',
    'heterochromia',
    'makeup', 'lipstick', 'mascara', 'eyeshadow',
    'braid', 'braids', 'bun', 'ponytail', 'ringlet', 'ringlets',
    'dreadlock', 'dreadlocks', 'drill',
    'face', 'head',
}

HAIR_HEAD_PARTIAL = [
    'twintail', 'pigtail',
    'hair clip', 'hair ornament', 'hair ribbon', 'hair bow', 'hair tie',
    'hair accessory', 'hair pin', 'hairpin', 'hair band', 'hair bun',
    'drill hair', 'antenna hair', 'hair streaks', 'hair intakes',
    'hair over', 'hair between', 'hair behind',
    'short hair', 'long hair', 'medium hair', 'very long hair',
    'straight hair', 'wavy hair', 'curly hair', 'messy hair',
    'blonde', 'silver hair', 'white hair', 'black hair', 'blue hair',
    'green hair', 'pink hair', 'purple hair', 'red hair', 'orange hair',
    'brown hair', 'grey hair', 'gray hair', 'multicolored hair', 'gradient hair',
    'blue eyes', 'green eyes', 'brown eyes', 'red eyes', 'purple eyes',
    'yellow eyes', 'pink eyes', 'grey eyes', 'gray eyes', 'teal eyes',
    'orange eyes', 'black eyes', 'glowing eyes', 'empty eyes',
    'closed eyes', 'half-closed eyes',
    'cat ears', 'fox ears', 'bunny ears', 'dog ears', 'wolf ears',
    'animal ears', 'pointy ears', 'elf ears', 'demon horns', 'angel wings',
    'ahoge',
]

# --- Clothing & Body ---

CLOTHING_BODY_WORDS = {
    'dress', 'shirt', 'skirt', 'jacket', 'coat', 'sweater', 'hoodie',
    'uniform', 'outfit', 'clothing', 'clothes',
    'pants', 'shorts', 'leggings',
    'apron', 'maid',
    'swimsuit', 'bikini',
    'leotard', 'lingerie', 'underwear', 'bra', 'panties', 'pantsu',
    'bodice', 'corset', 'bodysuit',
    'sleeve', 'sleeves', 'collar',
    'necktie', 'tie', 'scarf', 'cape', 'cloak',
    'armor', 'armour',
    'blouse', 'cardigan', 'vest', 'blazer', 'tuxedo',
    'kimono', 'yukata', 'serafuku',
    'robe', 'gown', 'toga', 'jumpsuit', 'overalls',
    'gloves', 'mittens',
    'belt', 'suspenders', 'choker',
    'navel', 'midriff', 'abdomen', 'belly',
    'breast', 'breasts', 'chest', 'cleavage', 'nipple', 'nipples',
    'collarbone', 'clavicle',
    'shoulder', 'shoulders',
    'waist', 'hip', 'hips',
    'skin', 'topless', 'bottomless', 'nude', 'naked',
    'arm', 'arms', 'torso',
}

CLOTHING_BODY_PARTIAL = [
    'crop top', 'tank top', 'tube top',
    'bare shoulders', 'bare arms', 'bare back', 'bare midriff',
    'open shirt', 'open jacket', 'torn clothes',
    'sailor collar', 'sailor uniform',
    'school uniform', 'military uniform',
    'maid outfit', 'maid uniform',
    'bowtie', 'bow tie', 'neck ribbon', 'neck tie',
    'see-through', 'see through', 'transparent',
    'latex', 'spandex', 'rubber',
    'cosplay', 'costume',
    'turtleneck',
    'thighhigh', 'thigh-high', 'kneehigh', 'knee-high',
]

# --- Feet & Shoes ---

FEET_SHOES_WORDS = {
    'feet', 'foot', 'barefoot',
    'shoes', 'boots', 'sandals', 'heels', 'loafers', 'sneakers', 'slippers',
    'socks', 'stockings', 'pantyhose', 'tights',
    'toes', 'ankle', 'ankles', 'calves', 'calf',
    'legs', 'leg', 'thigh', 'thighs', 'knee', 'knees',
}

FEET_SHOES_PARTIAL = [
    'high heels', 'platform shoes', 'mary jane', 'stiletto',
    'over-the-knee', 'knee-high socks', 'thigh-high socks',
    'bare feet', 'bare legs',
]


def _split_words(tag: str) -> set:
    return set(re.split(r'[\s_-]+', tag.lower()))


def _matches(tag: str, word_set: set, partial_list: list) -> bool:
    tag_lower = tag.lower()
    if _split_words(tag) & word_set:
        return True
    return any(p in tag_lower for p in partial_list)


def categorize_tags(tags: list) -> dict:
    """
    Categorize tags into LoRA-relevant groups.
    Priority: general > hair_head > clothing_body > feet_shoes > trigger
    Returns dict with sorted lists per category.
    """
    result = {
        'trigger': [],
        'hair_head': [],
        'clothing_body': [],
        'feet_shoes': [],
        'general': [],
    }

    for tag in tags:
        tag_lower = tag.lower()

        if tag_lower in GENERAL_EXACT or any(p in tag_lower for p in GENERAL_PARTIAL):
            result['general'].append(tag)
        elif _matches(tag, HAIR_HEAD_WORDS, HAIR_HEAD_PARTIAL):
            result['hair_head'].append(tag)
        elif _matches(tag, CLOTHING_BODY_WORDS, CLOTHING_BODY_PARTIAL):
            result['clothing_body'].append(tag)
        elif _matches(tag, FEET_SHOES_WORDS, FEET_SHOES_PARTIAL):
            result['feet_shoes'].append(tag)
        else:
            result['trigger'].append(tag)

    for key in result:
        result[key].sort()

    return result
