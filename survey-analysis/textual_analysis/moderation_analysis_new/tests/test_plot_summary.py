"""Check marginal percentages retain overlapping mechanisms and no-mechanism units."""
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fig_theory_mechanism_types_pre_post as plot


class PlotSummaryTests(unittest.TestCase):
    def test_overlapping_mechanisms_are_counted_independently(self):
        units = []
        for flags in itertools.product(('yes', 'no'), repeat=3):
            row = {'group': 'GenAI', 'phase': 'post'}
            row.update({f'{key}_present': flag for key, flag in zip(plot.PRESENCE_ORDER[:3], flags)})
            units.append(row)
        rows = [r for r in plot.summarize_presence(units) if r['group'] == 'GenAI' and r['phase'] == 'post']
        self.assertEqual([r['count'] for r in rows], [4, 4, 4, 1])
        self.assertEqual([r['n_pairs'] for r in rows], [8]*4)
        self.assertEqual([r['percentage'] for r in rows], [50., 50., 50., 12.5])

    def test_changes_use_unrounded_post_minus_pre(self):
        rows=[]
        for group in plot.GROUP_ORDER:
            for phase, count, n in [('pre',1,3),('post',2,4)]:
                for mechanism in plot.PRESENCE_ORDER:
                    rows.append({'group':group,'phase':phase,'mechanism':mechanism,
                                 'count':count,'n_pairs':n,'percentage':100*count/n})
        changes=plot.summarize_changes(rows)
        self.assertEqual(len(changes),12)
        for row in changes:
            self.assertAlmostEqual(row['change_percentage_points'],100*2/4-100*1/3)
            self.assertEqual((row['pre_n_pairs'],row['post_n_pairs']),(3,4))

    def test_exclusive_combinations_sum_to_one_hundred(self):
        units=[]
        for flags in itertools.product(('yes','no'), repeat=3):
            pair={key:{'present':flag} for key,flag in zip(plot.PRESENCE_ORDER[:3],flags)}
            units.append({'group':'GenAI','phase':'post','joint_type':plot.mechanism_combination(pair)})
        rows=[r for r in plot.summarize_combinations(units) if r['group']=='GenAI' and r['phase']=='post']
        self.assertEqual(len(rows),8)
        self.assertEqual(sum(r['count'] for r in rows),8)
        self.assertAlmostEqual(sum(r['percentage'] for r in rows),100.)

    def test_actual_data_counts_match(self):
        units = plot.load_units()
        rows = plot.summarize_presence(units)
        for group in plot.GROUP_ORDER:
            for phase in plot.PHASE_ORDER:
                sub = [u for u in units if u['group'] == group and u['phase'] == phase]
                selected = [r for r in rows if r['group'] == group and r['phase'] == phase]
                self.assertEqual(len(selected), 4)
                self.assertTrue(all(r['n_pairs'] == len(sub) for r in selected))
                self.assertEqual(selected[-1]['count'], sum(u['joint_type'] == 'no_mechanism' for u in sub))
                self.assertEqual(sum(r['count'] for r in plot.summarize_combinations(sub)), len(sub))


if __name__ == '__main__':
    unittest.main()
