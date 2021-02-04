import copy
import pandas as pd
import numpy as np

def merge_force(look_match, verbose=False):
    x = copy.deepcopy(look_match)
    out = {}
    if x.__contains__('implicit_merges') and x.__contains__('merges'):
        print('MergeForce: both implicit_merges and merges components found.  Using merges component.')
        m = x['merges']
    elif not x.__contains__('implicit_merges') and not x.__contains__('merges'):
        if verbose:
            print('\nNo merges found.  Returning object x unaltered.\n')
        return x
    elif not x.__contains__('merges'):
        m = x['implicit_merges']
    else:
        m = x['merges']

    out['grd_ob'] = x['grd_ob']
    out['grd_fo'] = x['grd_fo']
    #out['identifier_function'] = x['identifier_function']
    out['identifier_label'] = x['identifier_label']
    out['match_type'] = ['MergeForce', x['match_type']]
    out['match_message'] = ''.join((x['match_message'], " (merged) "))
    xdim = x['grd_ob'].shape

    if m == None:
        nmatches = 0
    else:
        nmatches = len(m)

    ar = np.arange(1, nmatches + 1, 1)
    matches = np.vstack((ar, ar)).T
    matches = pd.DataFrame(matches, columns = ["Forecast", "Observed"])
    out['matches'] = matches
    #xp = x['Xlabelsfeature']
    #yp = x['Ylabelsfeature']
    xp = x['grd_ob_features']
    yp = x['grd_fo_features']
    xfeats = {'Type': xp['Type'], 'xrange': xp['xrange'], 'yrange': xp['yrange'], 'dim': xp['dim'], 'warnings': xp['warnings'],
              'xstep': xp['xstep'], 'ystep': xp['ystep'], 'area': xp['area'], 'xcol': xp['xcol'], 'yrow': xp['yrow']}
    yfeats = {'Type': yp['Type'], 'xrange': yp['xrange'], 'yrange': yp['yrange'], 'dim': yp['dim'], 'warnings': yp['warnings'],
              'xstep': yp['xstep'], 'ystep': yp['ystep'], 'area': yp['area'], 'xcol': yp['xcol'], 'yrow': yp['yrow']}
    xlabeled = np.zeros([xdim[0], xdim[1]], dtype=int)
    ylabeled = np.zeros([xdim[0], xdim[1]], dtype=int)

    if verbose:
        print("Loop through ", nmatches, " components of merges list to set up new (matched) features.\n")
    for i in range(nmatches):
        if verbose:
            print(i, " ")
        tmp = np.array(m[i])
        uX = sorted(set(tmp[:, 1]))
        uY = sorted(set(tmp[:, 0]))
        nX = len(uX)
        nY = len(uY)
        xtmp = xp['labels_' + str(uX[0] + 1)]
        ytmp = yp['labels_' + str(uY[0] + 1)]
        if nX > 1:
            for j in range(1, nX):
                xtmp = xtmp | xp['labels_' + str(uX[j] + 1)]
        if nY > 1:
            for k in range(1, nY):
                ytmp = ytmp | yp['labels_' + str(uY[k] + 1)]
        xfeats['labels_' + str(i + 1)] = xtmp
        yfeats['labels_' + str(i + 1)] = ytmp
        nozero = np.transpose(np.nonzero(xtmp))
        for j in np.arange(len(nozero)):
            xlabeled[nozero[j][0]][nozero[j][1]] = i + 1
        nozero = np.transpose(np.nonzero(ytmp))
        for j in np.arange(len(nozero)):
            ylabeled[nozero[j][0]][nozero[j][1]] = i + 1
    if x['unmatched']['X'] == None or x['unmatched']['X'] == 'NULL':
        unX = x['unmatched']['X']
        nX2 = 0
    elif type(x['unmatched']['X']) == int:
        unX = x['unmatched']['X']
        nX2 = 1
    else:
        unX = sorted(x['unmatched']['X'])
        nX2 = len(unX)
    if x['unmatched']['Xhat'] == None or x['unmatched']['Xhat'] == 'NULL':
        unY = x['unmatched']['Xhat']
        nY2 = 0
    elif type(x['unmatched']['Xhat']) == int:
        unY = x['unmatched']['Xhat']
        nY2 = 1
    else:
        unY = sorted(x['unmatched']['Xhat'])
        nY2 = len(unY)
    if nX2 > 0:
        if verbose:
            print("\nLoop to add/re-label all unmatched observed features.\n")
        vxunmatched = list(range((nmatches + 1), (nmatches + nX2 + 1)))
        for i in range(nX2):
            xtmp = xp['labels_' + str(unX[i] + 1)]
            #xfeats[nmatches + i] = xtmp
            xfeats['labels_' + str(i + 1)] = xtmp
            [rows, cols] = xtmp.shape
            for xi in range(rows):
                for xj in range(cols):
                    if xtmp[xi, xj]:
                        #xlabeled[xi, xj] = nmatches + i + 1
                        xlabeled[xi, xj] = -1
    else:
        vxunmatched = 0
    if nY2 > 0:
        if verbose:
            print("\nLoop to add/re-label all unmatched forecast features.\n")
        fcunmatched = list(range((nmatches + 1), (nmatches + nY2 + 1)))
        for i in range(nY2):
            ytmp = yp['labels_' + str(unY[i] + 1)]
            yfeats['labels_' + str(i + 1)] = ytmp
            #yfeats[nmatches + i] = ytmp
            [rows, cols] = ytmp.shape
            for yi in range(rows):
                for yj in range(cols):
                    if ytmp[yi, yj]:
                        #ylabeled[yi, yj] = nmatches + i + 1
                        ylabeled[yi, yj] = -1
    else:
        fcunmatched = 0
    out['grd_ob_features'] = xfeats
    out['grd_fo_features'] = yfeats
    out['grd_ob_labeled'] = xlabeled
    out['grd_fo_labeled'] = ylabeled
    out['unmatched'] = {'X': vxunmatched, 'Xhat': fcunmatched}
    out['MergeForced'] = True
    return out

'''
if __name__ == '__main__':
    data = np.load(r"F:\Work\MODE\tra_test\centmatch\centmatchResult_PA3.npy", allow_pickle=True).tolist()
    look2 = merge_force(data)
    pyplot.imshow(look2['Xlabeled'])
    pyplot.colorbar()
    pyplot.figure(2)
    pyplot.imshow(look2['Ylabeled'])
    pyplot.colorbar()

    levels=np.linspace(0.5,np.max((look2['Xlabeled'], look2['Ylabeled'])) + 0.5,
                       np.max((look2['Xlabeled'], look2['Ylabeled']))+1)
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    ax = ax.flatten()
    im1 = ax[0].contourf(look2['Xlabeled'], cmap = 'jet', levels = levels)
    im2 = ax[1].contourf(look2['Ylabeled'], cmap = 'jet', levels = levels)
    fig.colorbar(im2, ax=[ax[0], ax[1]], fraction=0.03, pad=0.05)
    pyplot.show()
    print("hello")
'''
