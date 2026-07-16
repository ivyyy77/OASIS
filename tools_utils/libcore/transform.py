import numpy as np
import torch


def projectPointToPixel(cam, pt, with_Rt = True):
    if (with_Rt):
        pt = forwarkTransform(pt, cam.R, cam.t)

    if len(pt.shape) == 1:
        x = pt[0] / pt[2]
        y = pt[1] / pt[2]
    else:
        x = pt[:, 0] / pt[:, 2]
        y = pt[:, 1] / pt[:, 2]

    px = x * cam.fx + cam.cx
    py = y * cam.fy + cam.cy

    if len(pt.shape) == 1:
        return np.array([px, py])
    else:
        return np.stack([px, py], axis=1)

def unprojectPixelToPoint(cam, px, py, depth=None, with_Rt = True):
    x = (px - cam.cx) / cam.fx
    y = (py - cam.cy) / cam.fy

    if depth is None:
        depth = np.ones_like(px)

    if isinstance(px, np.ndarray):
        pt = np.stack([depth * x, depth * y, depth], axis=1)
    else:
        pt = np.array([depth * x, depth * y, depth])

    if (with_Rt):
        pt = backwardTransform(pt, cam.R, cam.t)

    return pt

def convertDepthToDisp(depth, baseline, focal):
    disp = baseline * focal / depth
    disp[depth < 0.1] = 0
    return disp

def convertDispToDepth(disp, baseline, focal):
    depth = baseline * focal / disp
    depth[disp < 0.01] = 0
    return depth

def forwarkTransform(pt, R, t):
    if len(pt.shape) == 1:
        return R.dot(pt) + t
    else:
        return (R.dot(pt.T) + t[:, None]).T

def backwardTransform(pt, R, t):
    if len(pt.shape) == 1:
        return R.transpose().dot((pt - t))
    else:
        return (pt - t).dot(R)

def calcVMap(cam, depth=None, with_Rt=True):
    h = cam.h
    w = cam.w

    xi = [i for i in range(w)]
    yi = [i for i in range(h)]
    xx, yy = np.meshgrid(xi, yi)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    if depth is not None:
        vmap = unprojectPixelToPoint(cam, xx, yy, depth[yy, xx], with_Rt=with_Rt)
        vmap[depth.reshape([-1]) < 0.1, :] = np.NaN
    else:
        vmap = unprojectPixelToPoint(cam, xx, yy, None, with_Rt=with_Rt)

    vmap = vmap.reshape([h, w, 3])
    return vmap

def calcNMap(vmap):
    h = vmap.shape[0]
    w = vmap.shape[1]
    nmap = np.zeros((h, w, 3))

    xi = [i for i in range(w)]
    yi = [i for i in range(h)]
    xx, yy = np.meshgrid(xi, yi)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    valid_idxs = (xx < w - 1) & (yy < h - 1)
    xx = xx[valid_idxs]
    yy = yy[valid_idxs]
    v00 = vmap[yy, xx]
    v01 = vmap[yy, xx + 1]
    v10 = vmap[yy + 1, xx]
    nmap[:] = np.NaN
    nmap[yy, xx] = np.cross((v01 - v00), (v00 - v10))

    return nmap

def calcVNMap(cam, depth):
    vmap = calcVMap(cam, depth)
    nmap = calcNMap(vmap)
    return vmap, nmap

def calcDistanceMap(cam, depth):
    vmap = calcVMap(cam, depth, with_Rt=False)
    dist_map = np.linalg.norm(vmap, axis=2)
    dist_map[np.isnan(vmap[:, :, 0])] = 0
    return dist_map

def lookAt(eye, center, top):
    z = center - eye
    x = np.cross(z, top)
    y = np.cross(z, x)

    R = np.eye(3)
    R[0,:] = (x / np.linalg.norm(x)).reshape(1,3)
    R[1,:] = (y / np.linalg.norm(y)).reshape(1,3)
    R[2,:] = (z / np.linalg.norm(z)).reshape(1,3)
    return R

def perspectiveFromCamera(cam, near=0.1, far=100.0):
    return np.array([
        [2 * cam.fx / cam.w,    0, (cam.w - 2 * cam.cx) / cam.w, 0],
        [0,   -2 * cam.fy / cam.h, (cam.h - 2 * cam.cy) / cam.h, 0],
        [0,    0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
        [0,    0,           -1,           0]
    ]).astype(np.float32)

def makeTransform(R, t):
    Rt = np.concatenate((R, t[:, None]), axis=1)
    return np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0).astype(np.float32)

def convertRtFromCV2GL(R, t):
    R_gl = np.array([
        [R[0, 0], -R[0, 1], -R[0, 2]],
        [-R[1, 0], R[1, 1], R[1, 2]],
        [-R[2, 0], R[2, 1], R[2, 2]]
    ]).astype(np.float32)

    t_gl = np.array([t[0], -t[1], -t[2]]).astype(np.float32)
    return R_gl, t_gl

def fitLineCenterDirectionToPoints(points):
    pt_mean = points.mean(axis=0)

    uu, dd, vv = np.linalg.svd(points - pt_mean)

    return pt_mean, vv[0]

def homogenize(v, dim=2):
    '''
    args:
        v: (B, N, C)
    return:
        (B, N, C+1)
    '''
    if dim == 2:
        return torch.cat([v, torch.ones_like(v[:,:,:1])], -1)
    elif dim == 1:
        return torch.cat([v, torch.ones_like(v[:,:1,:])], 1)
    else:
        raise NotImplementedError('unsupported homogenize dimension [%d]' % dim)
