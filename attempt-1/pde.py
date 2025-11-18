import torch

mse_loss = torch.nn.MSELoss()

def pde(x1, x2, net):
    """
    Compute Δ²u = (u_xx + u_yy)_xx + (u_xx + u_yy)_yy
    at the points (x1, x2) via automatic differentiation.
    """
    # 1) forward pass
    u = net(x1, x2)

    # 2) first derivatives
    # u_x, u_y = torch.autograd.grad(
    #     u.sum(), (x1, x2), create_graph=True
    # )
    u_x = torch.autograd.grad(u.sum(), x1, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), x2, create_graph=True)[0]

    # 3) second derivatives
    u_xx = torch.autograd.grad(u_x.sum(), x1, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), x2, create_graph=True)[0]

    # 4) Laplacian
    lap = u_xx + u_yy

    # 5) third derivatives of lap
    # lap_x, lap_y = torch.autograd.grad(
    #     lap.sum(), (x1, x2), create_graph=True
    # )
    lap_x = torch.autograd.grad(lap.sum(), x1, create_graph=True)[0]
    lap_y = torch.autograd.grad(lap.sum(), x2, create_graph=True)[0]

    # 6) fourth derivatives (bi‐Laplacian pieces)
    lap_xx = torch.autograd.grad(lap_x.sum(), x1, create_graph=True)[0]
    lap_yy = torch.autograd.grad(lap_y.sum(), x2, create_graph=True)[0]




    return lap_xx + lap_yy



def bdry(x1,x2,n1,n2,net):
    out = net(x1,x2)
    u_x = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x1, create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x.sum(),x2,create_graph=True)[0]

    u_y = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_yx = torch.autograd.grad(u_y.sum(),x1,create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(),x2,create_graph=True)[0]

    return out, u_xx*n1**2 + u_xy*n2*n1+u_yx*n1*n2 + u_yy*n2**2


def pdeloss(net,intx1,intx2,pdedata,bdx1,bdx2,nx1,nx2,bdrydata_diri,bdrydat_neumann,bw_diri,bw_neumann):
    bdx1  = bdx1.detach().requires_grad_(True)
    bdx2  = bdx2.detach().requires_grad_(True)
    pout = pde(intx1,intx2,net)
    #print("pout[50:150]=\n",pout[50:100])
    bout_diri,bout_neumann = bdry(bdx1,bdx2,nx1,nx2,net)
    pres = mse_loss(pout,pdedata)
    bres = bw_diri*mse_loss(bout_diri,bdrydata_diri) + bw_neumann*mse_loss(bout_neumann,bdrydat_neumann)
    diri_loss = mse_loss(bout_diri,bdrydata_diri)
    neumann_loss = mse_loss(bout_neumann,bdrydat_neumann)
      

    loss = pres + bres 

    return loss, pres, bres, diri_loss, neumann_loss




