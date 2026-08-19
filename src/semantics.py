"""Semantic understanding utilities for VisionNarrator.

Generated from notebooks/08_semantic_understanding.ipynb (sections 5-12 and 16): lexicons, entity / action /
attribute / scene / relationship extraction, structured representation, consistency checks and story-prompt helpers.
Notebook 08 is the readable, step-by-step source; this module exposes the same functions for later stages.
"""
from collections import Counter

import spacy

_NLP = None


def load_nlp(model_name="en_core_web_sm"):
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(model_name)
    return _NLP


PERSON_NOUNS = {'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'toddler', 'person', 'people', 'guy', 'lady', 'gentleman', 'player', 'worker', 'soldier', 'officer', 'police', 'policeman', 'firefighter', 'chef', 'cook', 'doctor', 'nurse', 'student', 'teacher', 'musician', 'singer', 'dancer', 'performer', 'artist', 'athlete', 'runner', 'cyclist', 'biker', 'skier', 'skateboarder', 'surfer', 'swimmer', 'climber', 'rider', 'driver', 'vendor', 'customer', 'shopper', 'crowd', 'couple', 'family', 'friend', 'team', 'group', 'adult', 'teenager', 'teen', 'youth', 'male', 'female', 'mother', 'father', 'mom', 'dad', 'son', 'daughter', 'brother', 'sister', 'wife', 'husband', 'bride', 'groom', 'audience'}

ANIMAL_NOUNS = {'dog', 'puppy', 'cat', 'kitten', 'horse', 'pony', 'cow', 'sheep', 'goat', 'pig', 'bird', 'duck', 'goose', 'chicken', 'elephant', 'lion', 'tiger', 'bear', 'monkey', 'deer', 'rabbit', 'squirrel', 'fish', 'seal', 'dolphin', 'whale', 'camel', 'donkey', 'zebra', 'giraffe', 'eagle', 'hawk', 'owl', 'pigeon', 'seagull', 'butterfly', 'insect', 'animal'}

CLOTHING_NOUNS = {'shirt', 't-shirt', 'tshirt', 'jacket', 'coat', 'sweater', 'sweatshirt', 'hoodie', 'vest', 'dress', 'skirt', 'pants', 'jeans', 'jean', 'shorts', 'suit', 'uniform', 'jersey', 'hat', 'cap', 'helmet', 'scarf', 'glove', 'gloves', 'boot', 'boots', 'shoe', 'shoes', 'sneaker', 'sneakers', 'sandal', 'sandals', 'sock', 'socks', 'tie', 'glasses', 'sunglasses', 'goggles', 'mask', 'apron', 'costume', 'outfit', 'clothing', 'clothes', 'top', 'tank', 'swimsuit', 'bikini', 'robe', 'backpack', 'bag', 'purse', 'headband', 'bandana', 'wetsuit', 'overalls', 'hood', 'beanie', 'blouse', 'trousers'}

COLLECTIVE_NOUNS = {'group', 'crowd', 'couple', 'pair', 'bunch', 'team', 'line', 'row', 'number', 'lot', 'herd', 'flock', 'pack'}

SCENE_NOUNS = {'beach': 'outdoor', 'sand': 'outdoor', 'ocean': 'outdoor', 'sea': 'outdoor', 'wave': 'outdoor', 'water': 'outdoor', 'lake': 'outdoor', 'river': 'outdoor', 'pool': 'outdoor', 'pond': 'outdoor', 'street': 'outdoor', 'road': 'outdoor', 'sidewalk': 'outdoor', 'path': 'outdoor', 'trail': 'outdoor', 'track': 'outdoor', 'park': 'outdoor', 'field': 'outdoor', 'grass': 'outdoor', 'lawn': 'outdoor', 'yard': 'outdoor', 'garden': 'outdoor', 'forest': 'outdoor', 'woods': 'outdoor', 'tree': 'outdoor', 'mountain': 'outdoor', 'hill': 'outdoor', 'rock': 'outdoor', 'cliff': 'outdoor', 'snow': 'outdoor', 'ice': 'outdoor', 'city': 'outdoor', 'town': 'outdoor', 'building': 'outdoor', 'bridge': 'outdoor', 'market': 'outdoor', 'playground': 'outdoor', 'stadium': 'outdoor', 'court': 'outdoor', 'parking': 'outdoor', 'lot': 'outdoor', 'sky': 'outdoor', 'dirt': 'outdoor', 'mud': 'outdoor', 'desert': 'outdoor', 'farm': 'outdoor', 'fountain': 'outdoor', 'kitchen': 'indoor', 'room': 'indoor', 'bedroom': 'indoor', 'bathroom': 'indoor', 'restaurant': 'indoor', 'bar': 'indoor', 'store': 'indoor', 'shop': 'indoor', 'office': 'indoor', 'gym': 'indoor', 'classroom': 'indoor', 'church': 'indoor', 'hallway': 'indoor', 'stage': 'indoor', 'library': 'indoor', 'hospital': 'indoor', 'kitchen': 'indoor', 'arena': 'indoor', 'house': 'unknown', 'home': 'unknown', 'school': 'unknown', 'station': 'unknown', 'airport': 'unknown'}

SURFACE_NOUNS = {'table', 'floor', 'wall', 'window', 'door', 'stairs', 'stair', 'step', 'bench', 'chair', 'couch', 'sofa', 'bed', 'counter', 'desk', 'tent', 'roof', 'ledge', 'fence', 'rail', 'railing', 'platform', 'porch', 'deck', 'balcony', 'ground'}

COLOR_WORDS = {'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'gray', 'grey', 'brown', 'tan', 'beige', 'gold', 'silver', 'dark', 'light', 'colorful', 'striped', 'plaid'}

SIZE_WORDS = {'big', 'large', 'small', 'little', 'tall', 'short', 'huge', 'tiny', 'long', 'giant'}

AGE_WORDS = {'young', 'old', 'elderly', 'teenage', 'adult', 'baby', 'middle-aged', 'senior'}

MATERIAL_WORDS = {'wooden', 'metal', 'plastic', 'stone', 'brick', 'glass', 'leather', 'concrete', 'paper', 'steel'}

NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

SPATIAL_PREPS = {'in', 'on', 'at', 'near', 'by', 'beside', 'next', 'under', 'over', 'above', 'below', 'behind', 'through', 'down', 'along', 'across', 'into', 'onto', 'inside', 'outside', 'around', 'past', 'toward', 'towards', 'up', 'against', 'between', 'among', 'off', 'from', 'atop', 'underneath', 'beneath', 'outside', 'front'}

CLOTHING_PREPS = {'in', 'with', 'wearing'}

TIME_WEATHER_CUES = {'snow', 'snowy', 'winter', 'sunset', 'sunrise', 'night', 'evening', 'dark', 'sunny', 'sun', 'rain', 'rainy', 'cloudy', 'fog', 'foggy', 'storm', 'summer', 'autumn', 'fall', 'spring', 'daytime'}

def classify_noun(lemma):
    if lemma in PERSON_NOUNS:
        return 'person'
    if lemma in ANIMAL_NOUNS:
        return 'animal'
    if lemma in CLOTHING_NOUNS:
        return 'clothing'
    if lemma in SCENE_NOUNS:
        return 'location'
    return 'object'

def attribute_category(word):
    if word in COLOR_WORDS:
        return 'color'
    if word in SIZE_WORDS:
        return 'size'
    if word in AGE_WORDS:
        return 'age'
    if word in MATERIAL_WORDS:
        return 'material'
    return 'other'

def empty_representation(image, source, caption):
    return {'image': image, 'source': source, 'caption': caption, 'entities': [], 'actions': [], 'attributes': [], 'scene': {}, 'relationships': []}

def clothing_of_person(token):
    if token.lemma_.lower() not in CLOTHING_NOUNS:
        return False
    return any((t.lemma_.lower() in PERSON_NOUNS for t in token.doc))

def entity_count(root):
    for child in root.children:
        if child.dep_ == 'nummod':
            if child.like_num and child.text.isdigit():
                return int(child.text)
            return NUMBER_WORDS.get(child.lower_, child.lower_)
        if child.dep_ == 'det' and child.lower_ in ('a', 'an', 'one'):
            return 1
    if root.tag_ in ('NNS', 'NNPS'):
        return 'several'
    return 1

def extract_entities(doc):
    entities, used = ([], set())
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ == 'PRON' or root.i in used or clothing_of_person(root):
            continue
        lemma = root.lemma_.lower()
        text = chunk.text
        count = entity_count(root)
        plural = root.tag_ in ('NNS', 'NNPS')
        if lemma in COLLECTIVE_NOUNS:
            of_prep = next((c for c in root.children if c.dep_ == 'prep' and c.lower_ == 'of'), None)
            member = next((c for c in of_prep.children if c.dep_ == 'pobj'), None) if of_prep is not None else None
            if member is not None:
                used.add(member.i)
                count = lemma
                text = doc[chunk.start:member.right_edge.i + 1].text
                root = member
                lemma = member.lemma_.lower()
                plural = True
        used.add(root.i)
        entities.append({'id': len(entities), 'text': text, 'lemma': lemma, 'type': classify_noun(lemma), 'count': count, 'plural': plural, 'token': root.i})
    return entities

def resolve_noun(token, entities):
    if token is None:
        return None
    by_token = {e['token']: e['lemma'] for e in entities}
    if token.i in by_token:
        return by_token[token.i]
    if token.lemma_.lower() in COLLECTIVE_NOUNS:
        of_prep = next((c for c in token.children if c.dep_ == 'prep' and c.lower_ == 'of'), None)
        member = next((c for c in of_prep.children if c.dep_ == 'pobj'), None) if of_prep is not None else None
        if member is not None:
            return member.lemma_.lower()
    return token.lemma_.lower()

def verb_subject(verb):
    subject = next((c for c in verb.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
    if subject is not None:
        return subject
    if verb.dep_ in ('acl', 'relcl') and verb.head.pos_ in ('NOUN', 'PROPN'):
        return verb.head
    if verb.dep_ in ('conj', 'xcomp', 'advcl') and verb.head.pos_ in ('VERB', 'AUX') and (verb.head is not verb):
        return verb_subject(verb.head)
    return None

def extract_actions(doc, entities):
    actions = []
    for token in doc:
        if token.pos_ != 'VERB' or token.lemma_.lower() in ('be', 'have'):
            continue
        subject = verb_subject(token)
        obj = next((c for c in token.children if c.dep_ in ('dobj', 'obj')), None)
        prep_objects = []
        for prep in (c for c in token.children if c.dep_ == 'prep'):
            for pobj in (c for c in prep.children if c.dep_ == 'pobj'):
                prep_objects.append([prep.lower_, resolve_noun(pobj, entities)])
        particle = next((c.lower_ for c in token.children if c.dep_ == 'prt'), None)
        actions.append({'verb': token.lemma_.lower() + (f' {particle}' if particle else ''), 'text': token.text, 'subject': resolve_noun(subject, entities), 'object': resolve_noun(obj, entities), 'prep_objects': prep_objects, 'passive': subject is not None and subject.dep_ == 'nsubjpass', 'token': token.i})
    return actions

def adjective_group(token):
    words = [token]
    words.extend((c for c in token.children if c.dep_ == 'conj' and c.pos_ == 'ADJ'))
    return [w.lower_ for w in words]

def modifiers_of(noun):
    values = []
    for child in noun.children:
        if child.dep_ == 'amod':
            values.extend(adjective_group(child))
        elif child.dep_ == 'compound':
            values.append(child.lower_)
    return values

def clothing_owner(token, entities):
    person_tokens = sorted((e['token'] for e in entities if e['type'] == 'person'))
    node = token
    for _ in range(6):
        if node.dep_ == 'conj' and node.head is not node:
            node = node.head
            continue
        if node.dep_ == 'pobj' and node.head.dep_ == 'prep' and (node.head.lower_ in CLOTHING_PREPS):
            owner = node.head.head
            if owner.pos_ in ('VERB', 'AUX'):
                owner = verb_subject(owner)
            if owner is not None and owner.i in person_tokens:
                return owner.i
            break
        if node.dep_ in ('dobj', 'obj') and node.head.lemma_.lower() == 'wear':
            owner = verb_subject(node.head)
            if owner is not None and owner.i in person_tokens:
                return owner.i
            break
        if node.i in person_tokens:
            return node.i
        break
    preceding = [t for t in person_tokens if t < token.i]
    if preceding:
        return preceding[-1]
    return person_tokens[0] if person_tokens else None

def extract_attributes(doc, entities):
    wearing_by_owner = {}
    for token in doc:
        if token.lemma_.lower() in CLOTHING_NOUNS and token.pos_ in ('NOUN', 'PROPN'):
            owner = clothing_owner(token, entities)
            if owner is not None:
                wearing_by_owner.setdefault(owner, []).append({'item': token.lower_, 'modifiers': modifiers_of(token)})
    attributes = []
    for entity in entities:
        root = doc[entity['token']]
        values = [{'value': v, 'category': attribute_category(v)} for v in modifiers_of(root)]
        wearing = wearing_by_owner.get(entity['token'], []) if entity['type'] == 'person' else []
        attributes.append({'entity_id': entity['id'], 'entity': entity['lemma'], 'attributes': values, 'wearing': wearing})
    return attributes

def extract_scene(doc, entities):
    locations, seen = ([], set())
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in SCENE_NOUNS and lemma not in SURFACE_NOUNS:
            continue
        if token.dep_ == 'pobj' and token.head.pos_ == 'ADP':
            prep = token.head.lower_
        elif token.dep_ in ('nsubj', 'ROOT'):
            prep = None
        else:
            continue
        if lemma in seen:
            continue
        seen.add(lemma)
        locations.append({'location': lemma, 'prep': prep, 'setting': SCENE_NOUNS.get(lemma, 'unknown')})
    settings = {loc['setting'] for loc in locations if loc['setting'] != 'unknown'}
    if settings == {'outdoor'}:
        setting = 'outdoor'
    elif settings == {'indoor'}:
        setting = 'indoor'
    elif len(settings) == 2:
        setting = 'mixed'
    else:
        setting = 'unknown'
    cues = sorted({token.lemma_.lower() for token in doc if token.lemma_.lower() in TIME_WEATHER_CUES})
    return {'locations': locations, 'setting': setting, 'cues': cues}

def extract_relationships(doc, entities, actions):
    relationships, seen = ([], set())

    def add(subject, relation, obj, kind):
        if subject is None or obj is None:
            return
        key = (subject, relation, obj)
        if key in seen:
            return
        seen.add(key)
        relationships.append({'subject': subject, 'relation': relation, 'object': obj, 'kind': kind})
    for action in actions:
        add(action['subject'], action['verb'], action['object'], 'action')
        for prep, obj in action['prep_objects']:
            add(action['subject'], f"{action['verb']} {prep}", obj, 'action')
    for entity in entities:
        root = doc[entity['token']]
        for prep in (c for c in root.children if c.dep_ == 'prep' and c.lower_ in SPATIAL_PREPS):
            for pobj in (c for c in prep.children if c.dep_ == 'pobj'):
                if pobj.lemma_.lower() in CLOTHING_NOUNS or prep.lower_ == 'of':
                    continue
                add(entity['lemma'], prep.lower_, resolve_noun(pobj, entities), 'spatial')
    return relationships

def build_representation(image, source, caption):
    doc = load_nlp()(caption)
    rep = empty_representation(image, source, caption)
    rep['entities'] = extract_entities(doc)
    rep['actions'] = extract_actions(doc, rep['entities'])
    rep['attributes'] = extract_attributes(doc, rep['entities'])
    rep['scene'] = extract_scene(doc, rep['entities'])
    rep['relationships'] = extract_relationships(doc, rep['entities'], rep['actions'])
    return rep

def entity_set(rep):
    return {e['lemma'] for e in rep['entities']}

def check_consistency(rep, greedy_rep):
    flags = []
    for entity in rep['entities']:
        if isinstance(entity['count'], int) and entity['count'] > 1 and (not entity['plural']):
            flags.append(f"count/plural mismatch: '{entity['text']}'")
    duplicates = [lemma for lemma, n in Counter((e['lemma'] for e in rep['entities'])).items() if n > 1]
    for lemma in duplicates:
        flags.append(f'repeated entity: {lemma}')
    if not rep['entities']:
        flags.append('no entities')
    if not rep['actions']:
        flags.append('no action')
    for action in rep['actions']:
        if action['subject'] is None:
            flags.append(f"verb without subject: '{action['text']}'")
    settings = {loc['setting'] for loc in rep['scene']['locations']}
    if {'indoor', 'outdoor'} <= settings:
        flags.append('indoor/outdoor conflict')
    words = rep['caption'].split()
    bigrams = list(zip(words, words[1:]))
    if len(bigrams) != len(set(bigrams)):
        flags.append('repeated bigram')
    if not rep['caption'].rstrip().endswith('.'):
        flags.append('no terminal period')
    beam_set, greedy_set = (entity_set(rep), entity_set(greedy_rep))
    union = beam_set | greedy_set
    jaccard = 1.0 if not union else len(beam_set & greedy_set) / len(union)
    if jaccard < 0.5:
        flags.append(f'greedy/beam disagreement (Jaccard {jaccard:.2f})')
    return (flags, round(jaccard, 3))

UNCERTAIN_BELOW = 0.3

NUMBER_TEXT = {1: 'a', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}

def count_text(entity):
    count = entity['count']
    if isinstance(count, int):
        return NUMBER_TEXT.get(count, str(count))
    if count in COLLECTIVE_NOUNS:
        return f'a {count} of'
    return str(count)

def entity_phrase(entity, rep):
    attrs = next((a for a in rep['attributes'] if a['entity_id'] == entity['id']), None)
    words = [a['value'] for a in attrs['attributes']] if attrs else []
    phrase = f"{count_text(entity)} {' '.join(words + [entity['lemma']])}".strip()
    if attrs and attrs['wearing']:
        phrase += ' wearing ' + ', '.join((' '.join(w['modifiers'] + [w['item']]) for w in attrs['wearing']))
    if entity.get('confidence') is not None and entity['confidence'] < UNCERTAIN_BELOW:
        phrase += ' (uncertain)'
    return phrase

def action_phrase(action):
    parts = [action['subject'] or 'someone', action['verb']]
    if action['object']:
        parts.append(action['object'])
    for prep, obj in action['prep_objects']:
        parts.extend([prep, str(obj)])
    phrase = ' '.join(parts)
    if action.get('confidence') is not None and action['confidence'] < UNCERTAIN_BELOW:
        phrase += ' (uncertain)'
    return phrase

def story_prompt(rep):
    characters = [entity_phrase(e, rep) for e in rep['entities'] if e['type'] in ('person', 'animal')]
    objects = [entity_phrase(e, rep) for e in rep['entities'] if e['type'] == 'object']
    scene = rep['scene']
    setting_bits = [scene['setting']] + [f"{(loc['prep'] + ' ' if loc['prep'] else '')}the {loc['location']}" for loc in scene['locations']] + scene['cues']
    setting_text = ', '.join((bit for bit in setting_bits if bit and bit != 'unknown')) or 'unspecified'
    actions_text = '; '.join((action_phrase(a) for a in rep['actions'])) or 'none identified'
    relations_text = '; '.join((f"{r['subject']} {r['relation']} {r['object']}" for r in rep['relationships'])) or 'none'
    lines = [f'Setting: {setting_text}', f"Characters: {'; '.join(characters) or 'none identified'}", f"Objects: {'; '.join(objects) or 'none identified'}", f'Actions: {actions_text}', f'Relations: {relations_text}', f"Caption: {rep['caption']}"]
    if rep['flags']:
        lines.append(f"Caveats: {'; '.join(rep['flags'])}")
    return '\n'.join(lines)

STORY_INSTRUCTION = 'Write a short story of 4-6 sentences about this scene. Use only the characters, objects, actions and setting listed; do not invent people or places that are not listed, and stay vague about anything marked uncertain.'

SEQUENCE_INSTRUCTION = 'Write one continuous story of 8-12 sentences that moves through these scenes in order. Keep characters consistent when the same kind of character appears again, and stay vague about anything marked uncertain.'
