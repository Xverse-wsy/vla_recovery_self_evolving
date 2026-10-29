"""CPU reference gradients for uneven data partitions, including 16 ranks."""
import argparse
import json
from pathlib import Path
import torch
from epoch_plan import EpochPlan, ExactEpochSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    plans, gradients, zero_rank_checks = 0, 0, 0
    for n in [1, 15, 17, 33]:
        x = torch.cos(torch.arange(n*2, dtype=torch.float64).reshape(n,2))
        y = torch.sin(torch.arange(n,dtype=torch.float64))
        for world in [1,2,16]:
            for micro in [1,2,4,8]:
                for accumulation in [1,3,4]:
                    plan = EpochPlan(n,world,micro,accumulation)
                    rank_indices = [list(ExactEpochSampler(n,world,r,seed=42,micro_batch_size=micro)) for r in range(world)]
                    assert sorted(i for ids in rank_indices for i in ids if i>=0)==list(range(n))
                    assert len({(len(ids)+micro-1)//micro for ids in rank_indices})==1
                    plans += 1
                    for offset in range(0,plan.batches_per_epoch,accumulation):
                        calls = plan.update_batches(offset)
                        denominator = plan.global_samples(offset,calls)
                        reference_indices = []
                        actual_gradient = torch.zeros(2,dtype=torch.float64)
                        for rank, ids in enumerate(rank_indices):
                            for batch in range(offset,offset+calls):
                                positions=ids[batch*micro:(batch+1)*micro]
                                valid=[i for i in positions if i>=0]
                                reference_indices.extend(valid)
                                if not valid:
                                    zero_rank_checks += 1
                                # A communication-only batch has one dummy example.
                                indexes=[max(0,i) for i in positions]
                                assert indexes
                                weight=torch.tensor([0.2,-0.1],dtype=torch.float64,requires_grad=True)
                                loss=((x[indexes]@weight-y[indexes])**2).mean()
                                # Runtime scales by world*valid*accum/denominator,
                                # upstream divides by accum, replica reduction by world.
                                scaled=loss*(world*len(valid)/denominator)
                                actual_gradient += torch.autograd.grad(scaled,weight)[0]/world
                        assert len(reference_indices)==denominator
                        weight=torch.tensor([0.2,-0.1],dtype=torch.float64,requires_grad=True)
                        loss=((x[reference_indices]@weight-y[reference_indices])**2).mean()
                        reference=torch.autograd.grad(loss,weight)[0]
                        torch.testing.assert_close(actual_gradient,reference,rtol=1e-12,atol=1e-12)
                        gradients += 1
    report=dict(state='complete',plans_checked=plans,optimizer_gradients_compared=gradients,
                zero_weight_rank_batches_checked=zero_rank_checks,
                tested_world_sizes=[1,2,16], real_samples_not_duplicated=True,
                exact_global_mean_gradients=True, gpu_used=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
