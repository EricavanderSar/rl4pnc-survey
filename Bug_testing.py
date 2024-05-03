import grid2op
import json
from lightsim2grid import LightSimBackend
import numpy as np
from tqdm import tqdm

env_name = "l2rpn_case14_sandbox"

env = grid2op.make(env_name, backend=LightSimBackend())
env_rev = grid2op.make(env_name, backend=LightSimBackend())
env_noshunt = grid2op.make(env_name, backend=LightSimBackend())

act_deact_shunt = env._helper_action_env({"shunt": {"set_bus": [(0, -1)]}})
act_shunt_produces_0 =env._helper_action_env({"shunt": {"shunt_q": [(0, 0.)]}})

with open("./data/action_spaces/l2rpn_case14_sandbox/medha.json", "r") as f:
    acts_ = json.load(f)
with open("./data/action_spaces/l2rpn_case14_sandbox/medha_reversed.json", "r") as f:
    acts_rev = json.load(f)

for act, act_rev in tqdm(zip(acts_, acts_rev), total=len(acts_)):
    obs = env.reset()
    obs_rev = env_rev.reset()
    obs_noshunt = env_noshunt.reset()

    act_glop = env.action_space(act)
    lines_expacted, subs_impact = act_glop.get_topological_impact(obs.line_status)
    if subs_impact[type(env).shunt_to_subid].any():
        # try action without shunt
        _, _, _, _ = env_noshunt.step(act_shunt_produces_0)
        obs_noshunt, _, _, _ = env_noshunt.step(act_glop)
        # try action normal
        obs, reward, done, info = env.step(act_glop)
        # try action reversed
        sub_act = int(np.arange(env.n_sub)[subs_impact])
        topo_act = act_glop.sub_set_bus[act_glop.sub_set_bus>0]
        print(f'Substation {sub_act}, Topo {topo_act} ')
        rev_topo = np.where(topo_act == 1, 2, 1)
        act_rev = env.action_space(
            {"set_bus": {"substations_id": [(sub_act, rev_topo)]}}
        )
        obs_rev, reward_rev, done_rev, info_rev = env_rev.step(act_rev)
        # print rho values
        print('rho no shunt: ', obs_noshunt.rho.max())
        print('rho normal ', obs.rho.max())
        print('rho reversed: ', obs_rev.rho.max())

    # _,_,_,_ = env_noshunt.step(act_shunt_produces_0)
    # obs_noshunt, _, _, _ = env_noshunt.step(act_glop)
    #
    # obs, reward, done, info = env.step(act_glop)
    # # _,_,_,_ = env_rev.step(act_deact_shunt)
    # obs_rev, reward_rev, done_rev, info_rev = env_rev.step(env.action_space(act_rev))
    # if act != act_rev:
    #     print('Normal Act is')
    #     print(env.action_space(act))
    #     print('Reversed act is')
    #     print(env.action_space(act_rev))
    #     print('Difference rho values: ', np.abs(obs.rho - obs_rev.rho).max())
    #     print('rho no shunt: ', obs_noshunt.rho.max())
    #     print('rho normal ', obs.rho.max())
    #     print('rho reversed: ', obs_rev.rho.max())
    # if np.abs(obs.rho - obs_rev.rho).max() >= 1e-6:
    #     raise RuntimeError("Issue for an action")
