"""Story generation utilities for VisionNarrator.

Class-based version of notebooks/09_story_generation.ipynb (sections 3, 5 and 9): prompt templates with a one-shot
example, a StoryGenerator around a local instruction-tuned LLM, and a Grounder that measures hallucinations against a
structured scene record and drives the repair pass. Notebook 09 is the readable, step-by-step source.
"""
import time
from collections import Counter

import torch

from src import semantics as sem

STORY_STRUCTURE = {
    "short_description": {"sentences": (1, 1), "max_words": 25, "max_new_tokens": 40, "sample": False},
    "detailed_explanation": {"sentences": (3, 5), "max_words": 120, "max_new_tokens": 130, "sample": False},
    "story": {"sentences": (4, 6), "max_words": 160, "max_new_tokens": 220, "sample": True},
}

SYSTEM_PROMPT = (
    "You write about a single photograph using only a structured description of it. "
    "Rules: mention only the characters, objects, actions and setting that are listed; never give anyone a name; "
    "do not add people, animals, objects, places, weather, time of day or events that are not listed; "
    "treat anything marked (uncertain) as unsure and describe it vaguely or leave it out; "
    "never write the words uncertain, unspecified, unknown, unclear or none identified - if no setting is given, simply do not describe the place at all; "
    "write plain, concrete English; return only the requested text with no title, list or explanation."
)

MODE_INSTRUCTIONS = {
    "short_description": "Write ONE sentence of at most 25 words that describes the photograph using the listed elements.",
    "detailed_explanation": "Write one factual paragraph of 3 to 5 sentences explaining what the photograph shows: who or what is present, what they look like, what they are doing and where. Do not tell a story and do not speculate beyond the list.",
    "story": "Write a short story of 4 to 6 sentences about this photograph: begin with the setting and the listed characters as they are, then show the listed actions happening, then end with what the characters do next without leaving the scene. You may add feelings, motives and small movements of the listed characters, but no new characters, animals, objects, places or names.",
}

EXAMPLE_PROMPT = (
    "Setting: outdoor, on the grass\n"
    "Characters: two children; a dog (uncertain)\n"
    "Objects: a red ball\n"
    "Actions: children play with ball; children run\n"
    "Relations: children play with ball; children on grass\n"
    "Caption: Two children are playing with a red ball on the grass ."
)
EXAMPLE_ANSWERS = {
    "short_description": "Two children play with a red ball on the grass.",
    "detailed_explanation": "Two children are on a patch of grass outdoors. They are playing with a red ball and running as they play. There may also be a dog nearby, though that is hard to tell. Nothing else about the place is known.",
    "story": "Two children raced across the grass with a red ball bouncing between them. The taller one kicked it hard, and the other sprinted after it, laughing. Something small moved at the edge of the grass, perhaps a dog, but the children were too busy to notice. When the ball finally rolled to a stop, they flopped down side by side to catch their breath. Then one of them stood, picked up the ball, and the game began again.",
}

EMPTY_VALUES = {"unspecified", "none identified", "none", "unknown"}

NEUTRAL_WORDS = {
    "day", "time", "moment", "morning", "afternoon", "evening", "hour", "minute", "second", "while", "way", "thing", "something",
    "someone", "everyone", "nothing", "anything", "life", "world", "place", "scene", "picture", "image", "photo", "photograph",
    "view", "feeling", "smile", "laugh", "laughter", "joy", "peace", "silence", "sound", "noise", "step", "pace", "glance", "look",
    "breath", "air", "warmth", "cold", "heat", "rest", "break", "journey", "walk", "ride", "trip", "game", "fun", "energy",
    "focus", "attention", "thought", "idea", "memory", "story", "end", "beginning", "start", "side", "top", "bottom", "front",
    "back", "middle", "edge", "distance", "direction", "spot", "part", "kind", "sort", "bit", "lot", "one", "other", "friendship",
    "company", "conversation", "word", "voice", "hand", "hands", "eye", "eyes", "face", "head", "arm", "arms", "leg", "legs", "foot",
    "feet", "body", "mind", "heart", "shoulder", "shoulders", "smiles", "moments", "steps", "words", "faces", "background",
}
SYNONYMS = {
    "people": {"friend", "friends", "folk", "folks", "stranger", "strangers", "passerby", "passersby", "pedestrian", "pedestrians", "adult", "adults", "individual", "individuals", "person", "crowd", "group", "everyone", "companion", "companions"},
    "man": {"guy", "gentleman", "fellow", "person", "adult", "he"},
    "woman": {"lady", "person", "adult", "she"},
    "boy": {"kid", "child", "youngster", "son"},
    "girl": {"kid", "child", "youngster", "daughter"},
    "child": {"kid", "boy", "girl", "youngster", "toddler"},
    "children": {"kids", "boys", "girls", "youngsters"},
    "dog": {"puppy", "pup", "canine", "pet"},
    "cat": {"kitten", "pet"},
    "horse": {"pony", "mare", "stallion"},
}
NAME_ENTITY_TYPES = {"PERSON", "GPE", "LOC", "ORG", "FAC", "NORP", "EVENT"}
ABSTRACT_SUFFIXES = ("ness", "tion", "sion", "ment", "ity", "ance", "ence", "ship", "hood", "ism", "ure", "age")
PLACE_NOUNS = set(sem.SCENE_NOUNS) | set(sem.SURFACE_NOUNS)
CHARACTER_NOUNS = sem.PERSON_NOUNS | sem.ANIMAL_NOUNS


def prompt_block(record):
    lines = []
    for line in record["prompt"].splitlines():
        key, _, value = line.partition(":")
        if value.strip().lower() in EMPTY_VALUES:
            continue
        lines.append(line)
    return "\n".join(lines)


def build_messages(record, mode, feedback=None):
    example_user = f"Photograph description:\n{EXAMPLE_PROMPT}\n\n{MODE_INSTRUCTIONS[mode]}"
    user = f"Photograph description:\n{prompt_block(record)}\n\n{MODE_INSTRUCTIONS[mode]}"
    if feedback:
        user += f"\n\nYour previous attempt broke the rules: {feedback} Rewrite the whole text and follow the rules strictly."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": EXAMPLE_ANSWERS[mode]},
        {"role": "user", "content": user},
    ]


class StoryGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device=None, temperature=0.6, top_p=0.9, seed=42, load=True):
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.tokenizer = None
        self.model = None
        self.available = False
        self.load_seconds = None
        self.error = None
        if load:
            self.load()

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()
            self.available = True
        except Exception as error:
            self.error = f"{type(error).__name__}: {str(error)[:200]}"
            self.available = False
        self.load_seconds = time.time() - start
        return self.available

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0

    @torch.no_grad()
    def generate_text(self, messages, max_new_tokens, sample=False, seed=None):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        n_prompt = inputs["input_ids"].shape[-1]
        torch.manual_seed(self.seed if seed is None else seed)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if sample:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            kwargs.update(do_sample=False)
        output = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(output[0, n_prompt:], skip_special_tokens=True).strip()

    def generate(self, record, mode, seed=None, feedback=None):
        if not self.available:
            return record["caption"] if mode == "short_description" else record["prompt"].replace("\n", " ")
        spec = STORY_STRUCTURE[mode]
        return self.generate_text(build_messages(record, mode, feedback), spec["max_new_tokens"], sample=spec["sample"], seed=seed)


class Grounder:
    def __init__(self, nlp=None, min_entity_coverage=0.5, max_repairs=1):
        self.nlp = nlp if nlp is not None else sem.load_nlp()
        self.min_entity_coverage = min_entity_coverage
        self.max_repairs = max_repairs

    def lemmas(self, text):
        return [t.lemma_.lower() for t in self.nlp(text) if not t.is_punct and not t.is_space]

    def allowed_vocabulary(self, record):
        s = record["semantics"]
        allowed = set(NEUTRAL_WORDS)
        for e in s["entities"]:
            allowed.add(e["lemma"])
            allowed |= SYNONYMS.get(e["lemma"], set())
        for attr in s["attributes"]:
            allowed |= {a["value"] for a in attr["attributes"]}
            for w in attr["wearing"]:
                allowed.add(w["item"])
                allowed |= set(w["modifiers"])
                allowed |= set(self.lemmas(w["item"]))
        for loc in s["scene"]["locations"]:
            allowed.add(loc["location"])
        allowed |= set(s["scene"]["cues"])
        for a in s["actions"]:
            allowed |= set(a["verb"].split())
            allowed |= {str(o) for _, o in a["prep_objects"]}
            if a["object"]:
                allowed.add(a["object"])
        allowed |= set(self.lemmas(record["caption"]))
        return allowed

    @staticmethod
    def is_abstract(token):
        lemma = token.lemma_.lower()
        return lemma.endswith(ABSTRACT_SUFFIXES) or (token.tag_ == "NN" and lemma.endswith("ing"))

    def report(self, text, record):
        doc = self.nlp(text)
        allowed = self.allowed_vocabulary(record)
        entity_lemmas = [e["lemma"] for e in record["semantics"]["entities"]]
        verb_lemmas = [a["verb"].split()[0] for a in record["semantics"]["actions"]]
        text_lemmas = {t.lemma_.lower() for t in doc if not t.is_punct}
        hallucinations, content_nouns, supported = [], 0, 0
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            lemma = token.lemma_.lower()
            is_name = token.text[:1].isupper() and not token.is_sent_start and (token.ent_type_ in NAME_ENTITY_TYPES or token.pos_ == "PROPN")
            if is_name:
                if lemma not in allowed and token.text.lower() not in allowed:
                    hallucinations.append({"word": token.text, "category": "name"})
                continue
            if token.pos_ != "NOUN":
                continue
            content_nouns += 1
            if lemma in allowed or token.text.lower() in allowed or self.is_abstract(token):
                supported += 1
                continue
            if lemma in CHARACTER_NOUNS:
                category = "new character"
            elif lemma in PLACE_NOUNS:
                category = "new place"
            else:
                category = "new detail"
            hallucinations.append({"word": token.text, "category": category})
        covered_entities = sum(1 for lemma in entity_lemmas if lemma in text_lemmas or (SYNONYMS.get(lemma, set()) & text_lemmas))
        covered_verbs = sum(1 for lemma in verb_lemmas if lemma in text_lemmas)
        counts = Counter(h["category"] for h in hallucinations)
        return {
            "content_nouns": content_nouns,
            "grounded_ratio": round(supported / content_nouns, 3) if content_nouns else 1.0,
            "entity_coverage": round(covered_entities / len(entity_lemmas), 3) if entity_lemmas else None,
            "action_coverage": round(covered_verbs / len(verb_lemmas), 3) if verb_lemmas else None,
            "hallucinations": hallucinations,
            "names": counts.get("name", 0),
            "new_characters": counts.get("new character", 0),
            "new_places": counts.get("new place", 0),
            "new_details": counts.get("new detail", 0),
            "sentences": len(list(doc.sents)),
            "words": sum(1 for t in doc if not t.is_punct and not t.is_space),
        }

    @staticmethod
    def violations(report):
        return report["names"] + report["new_characters"] + report["new_places"]

    def needs_repair(self, report):
        low_coverage = report["entity_coverage"] is not None and report["entity_coverage"] < self.min_entity_coverage
        return self.violations(report) > 0 or low_coverage

    @staticmethod
    def quality(report):
        return (-Grounder.violations(report), report["entity_coverage"] if report["entity_coverage"] is not None else 1.0, report["grounded_ratio"])

    def feedback_text(self, report, record):
        parts = []
        words = sorted({h["word"] for h in report["hallucinations"] if h["category"] in ("name", "new character", "new place")})
        if words:
            parts.append("you mentioned " + ", ".join(words) + ", which are not in the description")
        if report["entity_coverage"] is not None and report["entity_coverage"] < self.min_entity_coverage:
            parts.append("you left out listed elements: " + ", ".join(e["lemma"] for e in record["semantics"]["entities"]))
        return "; ".join(parts) + "."

    def check_and_repair(self, generator, record, mode, text, seed=42):
        report = self.report(text, record)
        repairs = 0
        while generator.available and self.needs_repair(report) and repairs < self.max_repairs:
            repairs += 1
            candidate = generator.generate(record, mode, seed=seed + 1000 * repairs, feedback=self.feedback_text(report, record))
            candidate_report = self.report(candidate, record)
            if self.quality(candidate_report) > self.quality(report):
                text, report = candidate, candidate_report
        report["repairs"] = repairs
        return text, report
