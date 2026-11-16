"""Offline checks for the expanded moderation prompt and response parsers."""
import copy
import json
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import coding_schema as schema
import code_moderation_pre_ML as pre
import code_moderation_post_ML as post


def mechanism_pair(features, **extra):
    return {
        **extra, "features": features, "discussed": "yes",
        "no_mechanism_reason": "",
        "causal_reconstruction": {"summary": "A stated process", "structure": []},
        "curvature": {"present": "no", "coding_rationale": "", "evidence": ""},
        "interaction_based": {
            "present": "yes", "interaction_location": "mediator_to_outcome",
            "coding_rationale": "The moderator changes the mediator's effect on Y.",
            "evidence": "motivation translates into a paper only with receptive outlets",
        },
        "other_mechanism": {"present": "no", "coding_rationale": "", "evidence": "", "outside_framework_reason": ""},
        "mechanism_implied_sign": "positive",
    }


def response(pairs):
    return {"pairs": pairs, "confidence": "high"}


class CodingSchemaTests(unittest.TestCase):
    def setUp(self):
        self.expected = [dict(feature_a=f"a{i}", feature_b=f"b{i}", predicted_sign="+") for i in range(3)]
        self.pairs = [mechanism_pair([p['feature_a'], p['feature_b']], rank=i+1)
                      for i, p in enumerate(self.expected)]

    def test_pre_accepts_all_locations(self):
        for location in schema.INTERACTION_LOCATIONS:
            with self.subTest(location=location):
                pairs = copy.deepcopy(self.pairs)
                pairs[0]['interaction_based']['interaction_location'] = location
                if location == 'not_applicable':
                    pairs[0]['interaction_based']['present'] = 'no'
                    pairs[0]['no_mechanism_reason'] = 'restatement'
                self.assertEqual(len(pre.extract_json(json.dumps(response(pairs)), self.expected)['pairs']), 3)

    def test_causal_reconstruction_is_preserved_per_pair(self):
        for module in (pre, post):
            expected, pairs = (self.expected, copy.deepcopy(self.pairs)) if module is pre else self.post_fixture()
            for i, pair in enumerate(pairs):
                pair['causal_reconstruction'] = {'summary': f'Pair {i} account', 'structure': [f'X1 -> M{i} -> Y']}
            result = module.extract_json(json.dumps(response(pairs)), expected)
            self.assertNotIn('causal_reconstruction', result)
            for i, pair in enumerate(result['pairs']):
                original = next(p for p in pairs if set(p['features']) == set(pair['features']))
                self.assertEqual(pair['causal_reconstruction'], original['causal_reconstruction'])

    def test_each_pair_requires_valid_causal_reconstruction(self):
        for module in (pre, post):
            for invalid in (None, {'summary': [], 'structure': []}, {'summary': '', 'structure': ''}):
                expected, pairs = (self.expected, copy.deepcopy(self.pairs)) if module is pre else self.post_fixture()
                if invalid is None:
                    del pairs[0]['causal_reconstruction']
                else:
                    pairs[0]['causal_reconstruction'] = invalid
                raw = response(pairs)
                raw['causal_reconstruction'] = {'summary': 'Top-level is insufficient', 'structure': []}
                with self.assertRaises(ValueError):
                    module.extract_json(json.dumps(raw), expected)

    def test_flat_sign_fields_in_saved_results(self):
        pre_pairs = pre.extract_json(json.dumps(response(self.pairs)), self.expected)['pairs']
        expected, pairs = self.post_fixture()
        post_pairs = post.extract_json(json.dumps(response(pairs)), expected)['pairs']
        for pair in pre_pairs + post_pairs:
            self.assertNotIn('interaction_sign', pair)
            self.assertEqual(pair['mechanism_implied_sign'], 'positive')
        self.assertTrue(all('sign_consistency' in pair for pair in pre_pairs))
        self.assertTrue(all('sign_consistency' in pair for pair in post_pairs))

    def test_saved_sign_fields_are_uniform_and_authoritative(self):
        obsolete = {'predicted_sign', 'ml_sign', 'user_predicted_sign', 'given_sign',
                    'consistent_with_predicted_sign', 'consistent_with_sign', 'interaction_sign'}
        for module in (pre, post):
            expected, pairs = (self.expected, copy.deepcopy(self.pairs)) if module is pre else self.post_fixture()
            for pair in pairs:
                pair.update({key: 'incorrect model output' for key in obsolete})
                pair['reference_sign'] = 'incorrect model output'
                pair['sign_consistency'] = 'incorrect model output'
            result = module.extract_json(json.dumps(response(pairs)), expected)['pairs']
            for pair, submitted in zip(result, expected):
                self.assertTrue(obsolete.isdisjoint(pair))
                wanted = (submitted['predicted_sign'] if module is pre else
                          submitted['user_predicted_sign'] if submitted['source'] == 'user_selected' else submitted['ml_sign'])
                self.assertEqual(pair['reference_sign'], wanted)
                self.assertEqual(pair['sign_consistency'], schema.sign_consistency(pair['mechanism_implied_sign'], wanted))

    def test_mechanism_judgments_allow_all_parallel_combinations(self):
        for states in product(('yes', 'no'), repeat=3):
            pair = copy.deepcopy(self.pairs[0])
            for key, state in zip(('curvature', 'interaction_based', 'other_mechanism'), states):
                pair[key]['present'] = state
            pair['interaction_based']['interaction_location'] = 'not_applicable' if states[1] == 'no' else 'unclear'
            if states[2] == 'yes':
                pair['other_mechanism'].update(coding_rationale='Alternative', evidence='Quote', outside_framework_reason='Outside both routes')
            if states == ('no', 'no', 'no'):
                pair['no_mechanism_reason'] = 'restatement'
            schema.validate_mechanism(pair)
            self.assertNotIn('primary_classification', pair)

    def test_location_matches_interaction_presence(self):
        pair = copy.deepcopy(self.pairs[0])
        pair['interaction_based']['interaction_location'] = 'not_applicable'
        with self.assertRaisesRegex(ValueError, 'cannot have interaction_location'):
            schema.validate_mechanism(pair)
        for location in schema.INTERACTION_LOCATIONS - {'not_applicable'}:
            pair['interaction_based'].update(present='no', interaction_location=location)
            pair['no_mechanism_reason'] = 'restatement'
            with self.assertRaisesRegex(ValueError, 'requires interaction_location'):
                schema.validate_mechanism(pair)

    def test_undiscussed_is_independent_of_missing_explanation(self):
        pair = copy.deepcopy(self.pairs[0])
        for key in ('curvature', 'interaction_based', 'other_mechanism'):
            pair[key]['present'] = 'no'
        pair['interaction_based']['interaction_location'] = 'not_applicable'
        for discussed in ('yes', 'no'):
            pair['discussed'] = discussed
            pair['no_mechanism_reason'] = '' if discussed == 'no' else 'restatement'
            schema.validate_mechanism(pair)
        pair['curvature']['present'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'undiscussed pair'):
            schema.validate_mechanism(pair)

    def test_no_mechanism_reason_conditions(self):
        for reason in ('restatement', 'unclear'):
            pairs = copy.deepcopy(self.pairs)
            for pair in pairs:
                for key in ('curvature', 'interaction_based', 'other_mechanism'):
                    pair[key]['present'] = 'no'
                pair['interaction_based']['interaction_location'] = 'not_applicable'
                pair['no_mechanism_reason'] = reason
                pair['discussed'] = 'yes'
            result = pre.extract_json(json.dumps(response(pairs)), self.expected)['pairs']
            self.assertTrue(all(pair['no_mechanism_reason'] == reason for pair in result))
            for invalid in ('', 'not_applicable', 'not_discussed', 'invalid'):
                pairs[0]['no_mechanism_reason'] = invalid
                with self.assertRaises(ValueError):
                    pre.extract_json(json.dumps(response(pairs)), self.expected)
        pair = copy.deepcopy(self.pairs[0])
        pair['no_mechanism_reason'] = 'restatement'
        with self.assertRaisesRegex(ValueError, "unless discussed"):
            schema.validate_mechanism(pair)

    def test_discussed_and_no_mechanism_reason_must_agree(self):
        pair = copy.deepcopy(self.pairs[0])
        for key in ('curvature', 'interaction_based', 'other_mechanism'):
            pair[key]['present'] = 'no'
        pair['interaction_based']['interaction_location'] = 'not_applicable'
        for discussed, reason in [('no', 'restatement'), ('no', 'unclear')]:
            pair.update(discussed=discussed, no_mechanism_reason=reason)
            with self.assertRaisesRegex(ValueError, "unless discussed"):
                schema.validate_mechanism(pair)

    def test_both_parsers_reject_location_conflicts(self):
        for module in (pre, post):
            expected, pairs = (self.expected, copy.deepcopy(self.pairs)) if module is pre else self.post_fixture()
            pairs[0]['interaction_based']['interaction_location'] = 'not_applicable'
            with self.assertRaisesRegex(ValueError, 'cannot have interaction_location'):
                module.extract_json(json.dumps(response(pairs)), expected)

    def test_mechanism_combination_retains_other_coexistence(self):
        combinations = set()
        for states in product(('yes', 'no'), repeat=3):
            pair = copy.deepcopy(self.pairs[0])
            for key, state in zip(('curvature', 'interaction_based', 'other_mechanism'), states):
                pair[key]['present'] = state
            combinations.add(schema.mechanism_combination(pair))
        self.assertEqual(len(combinations), 8)
        pair['other_mechanism']['present'] = 'unclear'
        with self.assertRaises(ValueError):
            schema.mechanism_combination(pair)

    def test_prompt_example_matches_parser_fields(self):
        for module in (pre, post):
            text = module.PROMPT_PATH.read_text()
            example, _ = json.JSONDecoder().raw_decode(text[text.index('\n{\n')+1:])
            required = {'rank' if module is pre else 'pair_id', *module.PAIR_FIELDS}
            self.assertEqual(set(example), required)
            self.assertNotIn('primary_classification', example)
            for name, fields in module.NESTED_FIELDS.items():
                self.assertEqual(set(example[name]), set(fields))

    def test_invalid_or_missing_location_rejected(self):
        for location in (None, 'none', 'other_chain_location', 'upstream'):
            pairs = copy.deepcopy(self.pairs)
            if location is None:
                del pairs[0]['interaction_based']['interaction_location']
            else:
                pairs[0]['interaction_based']['interaction_location'] = location
            with self.assertRaises(ValueError):
                pre.extract_json(json.dumps(response(pairs)), self.expected)

    def test_invalid_presence_and_sign_rejected(self):
        for field, value in [('mechanism_implied_sign', '+'), ('discussed', 'unclear')]:
            pair = copy.deepcopy(self.pairs[0]); pair[field] = value
            with self.assertRaises(ValueError):
                schema.validate_mechanism(pair)
        pair = copy.deepcopy(self.pairs[0])
        pair['curvature']['present'] = 'maybe'
        with self.assertRaises(ValueError):
            schema.validate_mechanism(pair)

    def test_unclear_presence_rejected_by_both_parsers(self):
        for module in (pre, post):
            for key in ('curvature', 'interaction_based', 'other_mechanism'):
                expected, pairs = (self.expected, copy.deepcopy(self.pairs)) if module is pre else self.post_fixture()
                pairs[0][key]['present'] = 'unclear'
                with self.assertRaisesRegex(ValueError, 'Invalid .*present'):
                    module.extract_json(json.dumps(response(pairs)), expected)

    def test_pre_preserves_input_features_and_sign(self):
        for field, value in [('features', ['wrong', 'pair'])]:
            pairs = copy.deepcopy(self.pairs); pairs[0][field] = value
            with self.assertRaises(ValueError):
                pre.extract_json(json.dumps(response(pairs)), self.expected)

    def post_fixture(self):
        user = [dict(p, user_rank=i+1) for i,p in enumerate(self.expected)]
        ml = [dict(feature_a=p['feature_b'], feature_b=p['feature_a'], ml_rank=i+1, ml_sign='-')
              for i,p in enumerate(self.expected)]
        expected = post.build_pairs_to_code(user, ml)
        pairs = []
        for e in post.unique_pairs_for_model(expected):
            pair = mechanism_pair([e['feature_a'], e['feature_b']], pair_id=e['pair_id'])
            pairs.append(pair)
        return expected, pairs

    def test_given_signs_are_not_sent_to_model(self):
        for module in (pre, post):
            tasks = module.TASKS
            template = module.load_prompt_template()
            theory = 'A neutral theory paragraph.'
            if module is pre:
                pairs = [dict(pair, rank=str(i+1)) for i, pair in enumerate(self.expected)]
            else:
                pairs, _ = self.post_fixture()
            feature_names = list(module.FEATURE_DEFINITIONS)
            for i, pair in enumerate(pairs):
                pair['feature_a'] = feature_names[(i % 3) * 2]
                pair['feature_b'] = feature_names[(i % 3) * 2 + 1]
            prompt = module.fill_prompt(template, tasks[0], pairs, theory)
            altered = copy.deepcopy(pairs)
            for pair in altered:
                for key in ('predicted_sign', 'user_predicted_sign', 'ml_sign'):
                    if pair.get(key) in ('+', '-'):
                        pair[key] = '-' if pair[key] == '+' else '+'
            self.assertEqual(prompt, module.fill_prompt(template, tasks[0], altered, theory))
            self.assertNotIn('predicted_sign', prompt)
            self.assertNotIn('ml_sign', prompt)

    def test_saved_signs_come_from_input_not_model(self):
        pairs = copy.deepcopy(self.pairs)
        pairs[0]['predicted_sign'] = '-'
        result = pre.extract_json(json.dumps(response(pairs)), self.expected)['pairs']
        self.assertEqual(result[0]['reference_sign'], '+')
        expected, pairs = self.post_fixture()
        pairs[0]['predicted_sign'] = '-'
        pairs[0]['ml_sign'] = '+'
        result = post.extract_json(json.dumps(response(pairs)), expected)['pairs']
        self.assertEqual(result[0]['reference_sign'], '+')
        self.assertEqual(result[3]['reference_sign'], '-')

    def test_post_overlap_keeps_separate_sign_judgments(self):
        expected, pairs = self.post_fixture()
        result = post.extract_json(json.dumps(response(pairs)), expected)['pairs']
        self.assertEqual(result[0]['sign_consistency'], 'yes')
        self.assertEqual(result[3]['sign_consistency'], 'no')
        self.assertEqual(result[3]['reference_sign'], '-')
        self.assertNotIn('predicted_sign', result[3])

    def test_pre_computes_all_sign_combinations(self):
        for implied in ('positive', 'negative', 'underdetermined'):
            for given in ('+', '-'):
                with self.subTest(implied=implied, given=given):
                    expected = copy.deepcopy(self.expected)
                    pairs = copy.deepcopy(self.pairs)
                    for e, pair in zip(expected, pairs):
                        e['predicted_sign'] = given
                        pair['mechanism_implied_sign'] = implied
                    result = pre.extract_json(json.dumps(response(pairs)), expected)['pairs']
                    wanted = ('underdetermined' if implied == 'underdetermined' else
                              'yes' if (implied == 'positive') == (given == '+') else 'no')
                    for pair in result:
                        self.assertEqual(pair['sign_consistency'], wanted)

    def test_post_derives_negative_and_underdetermined_signs(self):
        for implied, user_result, ml_result in [('negative', 'no', 'yes'),
                                                ('underdetermined', 'underdetermined', 'underdetermined')]:
            expected, pairs = self.post_fixture()
            for pair in pairs:
                pair['mechanism_implied_sign'] = implied
            result = post.extract_json(json.dumps(response(pairs)), expected)['pairs']
            self.assertEqual(result[0]['sign_consistency'], user_result)
            self.assertEqual(result[3]['sign_consistency'], ml_result)

    def test_model_supplied_consistency_is_overwritten(self):
        pairs = copy.deepcopy(self.pairs)
        pairs[0]['sign_consistency'] = 'no'
        result = pre.extract_json(json.dumps(response(pairs)), self.expected)['pairs']
        self.assertEqual(result[0]['sign_consistency'], 'yes')
        expected, pairs = self.post_fixture()
        pairs[0]['sign_consistency'] = 'yes'
        result = post.extract_json(json.dumps(response(pairs)), expected)['pairs']
        self.assertEqual(result[3]['sign_consistency'], 'no')

    def test_other_still_requires_its_own_evidence(self):
        pair = copy.deepcopy(self.pairs[0])
        pair['other_mechanism'].update(present='yes', coding_rationale='Alternative', outside_framework_reason='Outside both routes')
        with self.assertRaisesRegex(ValueError, 'requires coding_rationale, evidence'):
            schema.validate_mechanism(pair)

    def test_post_overlap_shares_coding_but_has_independent_saved_objects(self):
        expected, pairs = self.post_fixture()
        self.assertEqual(len(pairs), 3)
        result = post.extract_json(json.dumps(response(pairs)), expected)['pairs']
        self.assertEqual(len(result), 6)
        for a, b in zip(result[:3], result[3:]):
            for key in post.PAIR_FIELDS:
                if key != 'features':
                    self.assertEqual(a[key], b[key])
        result[0]['interaction_based']['coding_rationale'] = 'Changed in one saved slot'
        self.assertNotEqual(result[0]['interaction_based'], result[3]['interaction_based'])

    def test_post_handles_every_unique_pair_count(self):
        for unique_count in range(3, 7):
            user = [dict(p, user_rank=i+1) for i, p in enumerate(self.expected)]
            ml = []
            for i in range(3):
                if i < 6 - unique_count:
                    a, b = self.expected[i]['feature_b'], self.expected[i]['feature_a']
                else:
                    a, b = f'new_a{i}', f'new_b{i}'
                ml.append(dict(feature_a=a, feature_b=b, ml_rank=i+1, ml_sign='-'))
            slots = post.build_pairs_to_code(user, ml)
            unique = post.unique_pairs_for_model(slots)
            self.assertEqual(len(unique), unique_count)
            pairs = [mechanism_pair([e['feature_b'], e['feature_a']], pair_id=e['pair_id']) for e in unique]
            saved = post.extract_json(json.dumps(response(list(reversed(pairs)))), slots)['pairs']
            self.assertEqual([p['pair_id'] for p in saved], list(post.PAIR_IDS))
            self.assertEqual([p['reference_sign'] for p in saved], ['+']*3 + ['-']*3)

    def test_post_rejects_wrong_count_ids_and_features(self):
        expected, original = self.post_fixture()
        for change in ('missing', 'duplicate', 'unknown', 'features'):
            pairs = copy.deepcopy(original)
            if change == 'missing':
                pairs.pop()
            elif change == 'duplicate':
                pairs[1]['pair_id'] = pairs[0]['pair_id']
            elif change == 'unknown':
                pairs[0]['pair_id'] = 'pair_6'
            else:
                pairs[0]['features'] = ['wrong', 'pair']
            with self.assertRaises(ValueError):
                post.extract_json(json.dumps(response(pairs)), expected)

    def test_post_prompt_has_no_source_context_and_is_invariant_to_slot_order(self):
        slots, _ = self.post_fixture()
        names = list(post.FEATURE_DEFINITIONS)
        for i, slot in enumerate(slots):
            slot['feature_a'], slot['feature_b'] = names[2*(i%3):2*(i%3)+2]
        prompt = post.fill_prompt(post.load_prompt_template(), post.TASKS[0], slots, 'A neutral theory.')
        for forbidden in ('USER_SELECTED', 'ML_rank', 'Source:', 'post-ML', 'ML results', 'reference_sign'):
            self.assertNotIn(forbidden, prompt)
        self.assertEqual(prompt, post.fill_prompt(post.load_prompt_template(), post.TASKS[0], list(reversed(slots)), 'A neutral theory.'))

    def test_post_explicit_template_renders_only_existing_pairs(self):
        names = list(post.FEATURE_DEFINITIONS)
        for count in range(3, 7):
            user = [dict(feature_a=names[2*i], feature_b=names[2*i+1], user_rank=i+1, predicted_sign='+') for i in range(3)]
            ml = []
            for i in range(3):
                n = i if i < 6-count else i+3
                ml.append(dict(feature_a=names[2*n], feature_b=names[2*n+1], ml_rank=i+1, ml_sign='-'))
            slots = post.build_pairs_to_code(user, ml)
            unique = post.unique_pairs_for_model(slots)
            prompt = post.fill_prompt(post.load_prompt_template(), post.TASKS[0], slots, 'Neutral theory.')
            self.assertNotIn('{{', prompt)
            for i, pair in enumerate(unique, 1):
                a, b = pair['feature_a'], pair['feature_b']
                self.assertIn(f'pair_{i}. {a} × {b}', prompt)
                self.assertIn(f'{a}: {post.FEATURE_DEFINITIONS[a]}', prompt)
                self.assertIn(f'{b}: {post.FEATURE_DEFINITIONS[b]}', prompt)
            for i in range(count+1, 7):
                self.assertNotIn(f'pair_{i}.', prompt)

    def test_post_cannot_resume_results_from_previous_prompt_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'old_post.json'
            data = {'coding_schema': schema.CODING_SCHEMA, 'model': post.DEFAULT_MODEL,
                    'reasoning_effort': post.DEFAULT_REASONING_EFFORT,
                    'records': [{'coding': {'pairs': []}}]}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                post.load_output(path, post.DEFAULT_MODEL, post.DEFAULT_REASONING_EFFORT)

    def test_schema_is_saved_once_and_resume_still_works(self):
        for module in (pre, post):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'coding.json'
                data = module.load_output(path, module.DEFAULT_MODEL, module.DEFAULT_REASONING_EFFORT)
                data['records'] = [{'participant_id': 'test', 'task': 'racial_inequality',
                                    'coding_schema': schema.CODING_SCHEMA, 'coding': {'pairs': []}}]
                module.save_output(path, data)
                saved = module.load_output(path, module.DEFAULT_MODEL, module.DEFAULT_REASONING_EFFORT)
                self.assertEqual(saved['coding_schema'], schema.CODING_SCHEMA)
                self.assertNotIn('coding_schema', saved['records'][0])
                self.assertIn(('test', 'racial_inequality'), module.completed_keys(saved))
                self.assertEqual(len(schema.load_coding_records(path)), 1)

    def test_old_unused_reason_is_normalized_on_resume(self):
        for module in (pre, post):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'coding.json'
                data = module.load_output(path, module.DEFAULT_MODEL, module.DEFAULT_REASONING_EFFORT)
                pair = copy.deepcopy(self.pairs[0])
                pair['no_mechanism_reason'] = 'not_applicable'
                data['records'] = [{'coding': {'pairs': [pair]}}]
                path.write_text(json.dumps(data))
                loaded = module.load_output(path, module.DEFAULT_MODEL, module.DEFAULT_REASONING_EFFORT)
                self.assertEqual(loaded['records'][0]['coding']['pairs'][0]['no_mechanism_reason'], '')
                self.assertEqual(schema.load_coding_records(path)[0]['coding']['pairs'][0]['no_mechanism_reason'], '')
                self.assertEqual(pair['interaction_based']['interaction_location'], 'mediator_to_outcome')

    def test_old_schema_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'old.json'
            path.write_text(json.dumps(dict(coding_schema='mechanism_explanation_v3', records=[], model=pre.DEFAULT_MODEL, reasoning_effort=pre.DEFAULT_REASONING_EFFORT)))
            for module in (pre, post):
                with self.assertRaises(ValueError):
                    module.load_output(path, module.DEFAULT_MODEL, module.DEFAULT_REASONING_EFFORT)

    def test_full_input_preparation_without_api(self):
        for module in (pre, post):
            args = [module.load_rows(), module.TASKS, module.load_prompt_template()]
            if module is post:
                args.append(module.load_ml_top3())
            jobs = module.build_jobs(*args)
            self.assertEqual(len(jobs), 196)
            self.assertTrue(all('{{' not in job['prompt'] for job in jobs))
            self.assertTrue(all(len(job['pairs' if module is pre else 'pairs_to_code']) == (3 if module is pre else 6) for job in jobs))
            if module is post:
                self.assertTrue(all(3 <= len(job['model_pairs']) <= 6 for job in jobs))
                for job in jobs:
                    self.assertNotIn('USER_SELECTED', job['prompt'])
                    self.assertNotIn('Source:', job['prompt'])


if __name__ == '__main__':
    unittest.main()
