"""Unit tests of frozen beta optimizer, pilot extraction, and paired logger."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'e2_noisy_gradient' / 'src'))
import torch
from frozen_tensor_adam import FrozenTensorAdam, PilotCollector, beta_from_nv_state

class TestExp22(unittest.TestCase):
    def test_equal_to_adam_on_fixed_beta(self):
        p=torch.nn.Parameter(torch.tensor([1.,-2.]))
        q=torch.nn.Parameter(p.detach().clone())
        frozen=FrozenTensorAdam([('test',p)],{'test':.75}, lr=.001)
        adam=torch.optim.Adam([q], lr=.001,betas=(.75,.999))
        for _ in range(10):
            g=torch.tensor([.2,-.4]); p.grad=g.clone();q.grad=g.clone()
            frozen.step();adam.step()
        torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
    def test_beta_bounds(self):
        g={'response_mode':'distrust','gamma':1.,'beta_min':.1,'beta_base':.9}
        st={'rho':torch.tensor([0.,.5,1.])}
        torch.testing.assert_close(beta_from_nv_state(st,g),torch.tensor([.9,.5,.1]))
    def test_names_checked(self):
        with self.assertRaises(ValueError):
            FrozenTensorAdam([('a',torch.nn.Parameter(torch.ones(1)))], {'b':.8})
    def test_pilot_sample(self):
        p=torch.nn.Parameter(torch.ones(2));model=torch.nn.Linear(2,1,bias=False)
        model.weight=torch.nn.Parameter(torch.ones(1,2))
        class Dummy: pass
        opt=Dummy(); opt.param_groups=[{'params':[model.weight], 'response_mode':'distrust','gamma':1.,'beta_min':.1,'beta_base':.9}]
        opt.state={model.weight:{'rho':torch.tensor([[.1,.3]])}}
        collector=PilotCollector(model,opt,sample_every=2)
        collector.record_step();collector.record_step()
        self.assertEqual(collector.samples,1)
        self.assertAlmostEqual(collector.result()['weight'], .74,places=6)

if __name__=='__main__':unittest.main()
