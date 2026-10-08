import unittest
from repro_1p7b.graph_frontier.verified_recovery_metrics import paired_statistics,state_difference


class RecoveryMetricsTest(unittest.TestCase):
    def test_task_paired_bootstrap_has_known_positive_difference(self):
        rows=[]
        for i in range(12):
            a={'calls':2,'corrections':1,'deep_ge2':0,'new_verified_states':0,'corrections_per100calls':50.,
                'deep_per100calls':0.,'new_states_per100calls':0.,'max_frontier':1,'new_student_reached_deeper_states':0}
            b={'calls':2,'corrections':2,'deep_ge2':1,'new_verified_states':1,'corrections_per100calls':100.,
                'deep_per100calls':50.,'new_states_per100calls':50.,'max_frontier':3,'new_student_reached_deeper_states':1}
            rows.append({'task_id':str(i),'arms':{'FD':a,'RECOVERY':b}})
        v=paired_statistics(rows,repeats=100)
        self.assertEqual(v['corrections_per100calls']['pooled_rate_difference_CI95'],[50.,50.])
        self.assertEqual(v['max_frontier']['win'],12)
        self.assertEqual(v['new_student_reached_deeper_states']['paired_mean_difference'],1.)

    def test_teacher_only_new_state_is_not_student_novelty(self):
        a={'verified_states':[{'state_hash':'initial','pointer':0,'origin':'INITIAL'}],'max_verified_frontier_depth':0}
        b={'verified_states':[{'state_hash':'new','pointer':1,'origin':'GOLD_LOCAL_CORRECTION'},
                              {'state_hash':'new','pointer':2,'origin':'STUDENT'}]}
        total,student,deeper,_=state_difference(a,b)
        self.assertEqual((total,student,deeper),(1,0,0))

    def test_same_physical_state_new_pointer_is_not_new_state(self):
        a={'verified_states':[{'state_hash':'same','pointer':0,'origin':'INITIAL'}],'max_verified_frontier_depth':0}
        b={'verified_states':[{'state_hash':'same','pointer':3,'origin':'STUDENT'}]}
        self.assertEqual(state_difference(a,b)[:3],(0,0,0))


if __name__=='__main__':unittest.main()
