import csv
import numpy as np
from ..plot.common_multi import *
from ..plot.common_multi import _traj_beta_inf_loop

Actions = GridWorldMDP.Actions

def velocity2Action(s1, s2):
    if s2 is None:
        return Actions.ABSORB
    elif s1[0]+1==s2[0] and s1[1]==s2[1]:
        return Actions.RIGHT
    elif s1[0]-1==s2[0] and s1[1]==s2[1]:
        return Actions.LEFT
    elif s1[0]==s2[0] and s1[1]+1==s2[1]:
        return Actions.UP
    elif s1[0]==s2[0] and s1[1]-1==s2[1]:
        return Actions.DOWN
    elif s1[0]+1==s2[0] and s1[1]+1==s2[1]:
        return Actions.UP_RIGHT
    elif s1[0]-1==s2[0] and s1[1]+1==s2[1]:
        return Actions.UP_LEFT
    elif s1[0]+1==s2[0] and s1[1]-1==s2[1]:
        return Actions.DOWN_RIGHT
    elif s1[0]-1==s2[0] and s1[1]-1==s2[1]:
        return Actions.DOWN_LEFT
    elif s1[0]==s2[0] and s1[1]==s2[1]:
        return Actions.ABSORB
    else:
        print (s2[0]-s1[0],s2[1]-s1[1])
        return Actions.ABSORB
        #raise Exception('Invalid velocity')

def filterNoise(data):
    def isNoise(s1, s2):
        return abs(s2[0]-s1[0])>1 or abs(s2[1]-s1[1])>1
    result = []
    for i, row in enumerate(data):
        if not result or not isNoise(row, result[-1]):
            result.append(row)
    return np.array(result)

def showProbs(start, dest_list, traj, g, T, R=-1,
        epsilon=0.05,
        title=None, inf_mod=inf_default, zmin=-5, zmax=0, traj_len=None,
        hmm=False, landmarks=[], **kwargs):
    beta_hat = beta_fixed = 1

    occ = inf_mod.occupancy
    def on_loop(traj, D, D_dest_list, dest_probs, betas, t):
        occ_list = list(D_dest_list)
        subplot_titles = []
        stars_grid = landmarks[:]

        for dest, dest_prob, beta_hat in zip(dest_list, dest_probs, betas):
            subplot_titles.append("dest_prob={}, beta_hat={}".format(
                dest_prob, beta_hat))
            stars_grid.append([dest])

        # Weighted average of all heat maps
        occ_list.append(D)
        subplot_titles.append("net occupancy")
        stars_grid.append(dest_list)

        _title = title or "euclid expected occupancies t={t} R={R}"
        _title = _title.format(T=T, t=t, R=R, traj="\{traj_mode\}", epsilon=epsilon)

        plot_heat_maps(g, traj, occ_list, subplot_titles, title=_title,
                stars_grid=stars_grid, zmin=zmin, zmax=zmax, **kwargs)

    _traj_beta_inf_loop(on_loop, g, traj, dest_list, traj_len=traj_len,
            hmm=hmm, hmm_opts=dict(epsilon=epsilon))

def run():
    with open('./pp/test/tanslation.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        data = []
        for row in spamreader:
            data.append(np.array([float(x) for x in row[1:-1]]))
        data = np.array(data)
        data = (np.round(data*5) + 20)
    print data

    traj = []
    #data = data[100:181,:]
    N = 50
    T = N+N
    R = -1
    g = GridWorldMDP(N, N, default_reward=R, allow_wait=True)
    landmarks = [g.coor_to_state(14,21),g.coor_to_state(14,13),g.coor_to_state(14,5),
        g.coor_to_state(26,21),g.coor_to_state(26,13),g.coor_to_state(26,5)]

    data = filterNoise(data)
    for i, row in enumerate(data):
        s1 = row
        s2 = None
        if i < len(data)-1:
            s2 = data[i+1]
        traj.append((int(g.coor_to_state(row[0], row[1])), velocity2Action(s1,s2)))

    # plot whole traj
    #plot_heat_maps(g, traj, [np.array([0]*2500)], [""], title="",
    #                stars_grid=landmarks, zmin=-5, zmax=0)

    start = int(g.coor_to_state(data[0,0], data[0,1]))
    # dest = int(g.coor_to_state(data[-1,0], data[-1,1]))
    showProbs(start, landmarks, traj, g, T, R, auto_open=False, last_only=True)
    print('Finished')
