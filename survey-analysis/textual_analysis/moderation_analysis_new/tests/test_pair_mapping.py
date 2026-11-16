"""Independent source-to-result audit of neutral post pair mapping (no API calls)."""
import copy
import csv
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import code_moderation_post_ML as post
import code_moderation_pre_ML as pre
from test_coding_schema import mechanism_pair


def key(features):
    return tuple(sorted(features))


def model_response(slots, seed):
    # Deliberately construct an independent oracle, rather than reuse the dedup helper.
    feature_pairs = sorted({key((s['feature_a'], s['feature_b'])) for s in slots})
    oracle = {}
    pairs = []
    for i, features in enumerate(feature_pairs):
        marker = ' | '.join(features)
        pair = mechanism_pair(list(reversed(features)), pair_id=f'pair_{i+1}')
        pair['causal_reconstruction'] = {'summary': marker, 'structure': [marker]}
        pair['interaction_based']['coding_rationale'] = marker
        pair['interaction_based']['evidence'] = marker
        pair['mechanism_implied_sign'] = ('positive', 'negative', 'underdetermined')[i % 3]
        oracle[features] = copy.deepcopy(pair)
        pairs.append(pair)
    random.Random(seed).shuffle(pairs)
    return {'confidence': 'high', 'pairs': pairs}, oracle


class MappingAuditTests(unittest.TestCase):
    def assert_mapping(self, slots, result, oracle):
        self.assertEqual(len(result), 6)
        self.assertEqual([p['pair_id'] for p in result], list(post.PAIR_IDS))
        for slot, saved in zip(slots, result):
            features = [slot['feature_a'], slot['feature_b']]
            source = 'user_selected' if slot['pair_id'].startswith('USER_SELECTED') else 'ml_selected'
            rank = int(slot['pair_id'].rsplit('_', 1)[1])
            reference = slot['user_predicted_sign'] if source == 'user_selected' else slot['ml_sign']
            self.assertEqual(saved['features'], features)
            self.assertEqual(saved['source'], source)
            self.assertEqual(saved['user_rank'], rank if source == 'user_selected' else None)
            self.assertEqual(saved['ml_rank'], rank if source == 'ml_selected' else None)
            self.assertEqual(saved['reference_sign'], reference)
            original = oracle[key(features)]
            for field in post.PAIR_FIELDS:
                if field != 'features':
                    self.assertEqual(saved[field], original[field])
            implied = original['mechanism_implied_sign']
            expected_consistency = ('underdetermined' if implied == 'underdetermined' else
                                    'yes' if (implied == 'positive') == (reference == '+') else 'no')
            self.assertEqual(saved['sign_consistency'], expected_consistency)

    def test_every_real_post_job_against_original_csv_and_ml_json(self):
        # Read original source columns directly, separately from the production loaders.
        with post.CSV_PATH.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        ml_raw = json.loads(post.ML_PATH.read_text())
        jobs = post.build_jobs(post.load_rows(), post.TASKS, post.load_prompt_template(), post.load_ml_top3())
        indexed = {(j['participant_id'], j['task']['task']): j for j in jobs}
        self.assertEqual(len(indexed), 196)
        unique_counts = set()
        records = []
        for row in rows:
            for prefix, task, ml_key in [('Race', 'racial_inequality', 'race'),
                                         ('Gender', 'gender_inequality', 'gender')]:
                job = indexed[(row['Participant ID'].strip(), task)]
                direct = []
                for rank, suffix in enumerate(('1st', '2nd', '3rd'), 1):
                    a, b = [x.strip() for x in row[f'Q {prefix}.{rank+5} (SOI, {suffix})'].split(',')]
                    sign = row[f'Q {prefix}.9 (SOI, sign, {suffix})'].strip()
                    direct.append(dict(pair_id=f'USER_SELECTED_rank_{rank}', feature_a=a, feature_b=b,
                                       user_predicted_sign=sign, ml_sign=None))
                for item in sorted(ml_raw[ml_key], key=lambda x: int(x['rank']))[:3]:
                    direct.append(dict(pair_id=f"ML_rank_{int(item['rank'])}",
                                       feature_a=item['feature_1'], feature_b=item['feature_2'],
                                       ml_sign=str(item['sign']).strip(), user_predicted_sign=None))
                for original, slot in zip(direct, job['pairs_to_code']):
                    for field, value in original.items():
                        self.assertEqual(slot[field], value)
                self.assertEqual(job['theory'], row[f'Q {prefix}.15 LLM_refined post-ML theory (SOI)'].strip())
                frozen = copy.deepcopy(job['pairs_to_code'])
                response, oracle = model_response(direct, len(records))
                unique_counts.add(len(response['pairs']))
                self.assertEqual(len(job['model_pairs']), len(response['pairs']))
                saved = post.extract_json(json.dumps(response), job['pairs_to_code'])
                self.assertEqual(job['pairs_to_code'], frozen)
                self.assert_mapping(direct, saved['pairs'], oracle)
                records.append({'participant_id': job['participant_id'], 'task': task, 'coding': saved})
        self.assertEqual(unique_counts, {3, 4, 5, 6})
        # Exercise the same atomic writer and loader used by the coding run.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'post.json'
            data = post.load_output(path, post.DEFAULT_MODEL, post.DEFAULT_REASONING_EFFORT)
            data['records'] = records
            post.save_output(path, data)
            restored = post.load_output(path, post.DEFAULT_MODEL, post.DEFAULT_REASONING_EFFORT)
            self.assertEqual(restored['records'], records)
            self.assertEqual(len(post.completed_keys(restored)), 196)

    def test_overlap_patterns_all_reference_signs_and_response_order(self):
        # 20 ML pair selections x 64 reference-sign combinations = 1,280 mappings.
        pool = [(f'a{i}', f'b{i}') for i in range(6)]
        for ml_selection in itertools.combinations(range(6), 3):
            for signs in itertools.product(('+', '-'), repeat=6):
                user = [dict(feature_a=a, feature_b=b, user_rank=i+1, predicted_sign=signs[i])
                        for i, (a, b) in enumerate(pool[:3])]
                ml = [dict(feature_a=pool[n][1], feature_b=pool[n][0], ml_rank=i+1, ml_sign=signs[i+3])
                      for i, n in enumerate(ml_selection)]
                slots = post.build_pairs_to_code(user, ml)
                response, oracle = model_response(slots, sum(ml_selection))
                result = post.extract_json(json.dumps(response), slots)['pairs']
                self.assert_mapping(slots, result, oracle)

    def test_swapped_ids_rejected_even_when_all_features_are_valid(self):
        user = [dict(feature_a=f'a{i}', feature_b=f'b{i}', user_rank=i+1, predicted_sign='+') for i in range(3)]
        ml = [dict(feature_a=f'a{i}', feature_b=f'b{i}', ml_rank=i+1, ml_sign='-') for i in range(3)]
        slots = post.build_pairs_to_code(user, ml)
        response, _ = model_response(slots, 0)
        first, second = response['pairs'][:2]
        first['pair_id'], second['pair_id'] = second['pair_id'], first['pair_id']
        with self.assertRaisesRegex(ValueError, 'features do not match'):
            post.extract_json(json.dumps(response), slots)

    def test_all_real_pre_signs_and_features_are_preserved(self):
        jobs = pre.build_jobs(pre.load_rows(), pre.TASKS, pre.load_prompt_template())
        for job in jobs:
            pairs = [mechanism_pair([e['feature_b'], e['feature_a']], rank=e['rank']) for e in job['pairs']]
            result = pre.extract_json(json.dumps({'confidence': 'high', 'pairs': list(reversed(pairs))}), job['pairs'])
            for original, saved in zip(job['pairs'], result['pairs']):
                self.assertEqual(saved['rank'], original['rank'])
                self.assertEqual(key(saved['features']), key((original['feature_a'], original['feature_b'])))
                self.assertEqual(saved['reference_sign'], original['predicted_sign'])


if __name__ == '__main__':
    unittest.main()
