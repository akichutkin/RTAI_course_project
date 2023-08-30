import argparse
import csv
import torch
from networks import get_network, get_net_name, NormalizedResnet
from deep_poly_network import *

DEVICE = 'cpu'
DTYPE = torch.float32

def transform_image(pixel_values, input_dim):
    normalized_pixel_values = torch.tensor([float(p) / 255.0 for p in pixel_values])
    if len(input_dim) > 1:
        input_dim_in_hwc = (input_dim[1], input_dim[2], input_dim[0])
        image_in_hwc = normalized_pixel_values.view(input_dim_in_hwc)
        image_in_chw = image_in_hwc.permute(2, 0, 1)
        image = image_in_chw
    else:
        image = normalized_pixel_values

    assert (image >= 0).all()
    assert (image <= 1).all()
    return image

def get_spec(spec, dataset):
    input_dim = [1, 28, 28] if dataset == 'mnist' else [3, 32, 32]
    eps = float(spec[:-4].split('/')[-1].split('_')[-1])
    test_file = open(spec, "r")
    test_instances = csv.reader(test_file, delimiter=",")
    for i, (label, *pixel_values) in enumerate(test_instances):
        inputs = transform_image(pixel_values, input_dim)
        inputs = inputs.to(DEVICE).to(dtype=DTYPE)
        true_label = int(label)
    inputs = inputs.unsqueeze(0)
    return inputs, true_label, eps


def get_net(net, net_name):
    net = get_network(DEVICE, net)
    state_dict = torch.load('../nets/%s' % net_name, map_location=torch.device(DEVICE))
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict)
    net = net.to(dtype=DTYPE)
    net.eval()
    if 'resnet' in net_name:
        net = NormalizedResnet(DEVICE, net)
    return net


def loss_fn(ub: torch.tensor):
    return torch.log(torch.max(ub))


def analyze(net, inputs, eps, true_label, dataset):
    #Depending on Net Type, need to choose different network Builders
    if type(net) == NormalizedResnet:
        DP_net = resnet_builder(net, inputs, true_label)
    else: 
        DP_net = network_builder(net, inputs, true_label)

    #Initiating our DP_object, which is to be pushed through the DeepPoly Network
    x = DP_object(inputs, eps, dataset=dataset)
    lb, ub = DP_net(x)

    #Optimization
    opt = torch.optim.Adam(DP_net.parameters(), lr=2)
    epoch = 0
    while True:
        #We will let this optimization run until time-out (1min). As soon as all ub < 0 we can return verified. 
        if torch.all(ub<0):
            return True
        epoch +=1

        #Need to instantiate deep poly object anew, because lb and ub lists need to be freed
        x = DP_object(inputs, eps, dataset=dataset)
        opt.zero_grad()
        loss = loss_fn(ub)
        loss.backward()
        opt.step()
        lb, ub = DP_net(x)
    


def error_check(net, inputs, true_label, dataset):
    error = DP_object(inputs, eps = 0, dataset= dataset)
    if type(net) == NormalizedResnet:
        DP_net = resnet_builder(net,inputs,true_label, error_check=True)
    else: 
        DP_net = network_builder(net, inputs, true_label, error_check=True)
    DP_net(error)
    error.backsub()
    lb, ub = error.abs_bounds[-1]
    check = torch.all(torch.isclose(lb.reshape(1,-1), net(inputs)))
    print("error_check successful: ", check)
    


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net', type=str, required=True, help='Neural network architecture to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    net_name = get_net_name(args.net)
    dataset = 'mnist' if 'mnist' in net_name else 'cifar10'
    
    inputs, true_label, eps = get_spec(args.spec, dataset)
    net = get_net(args.net, net_name)

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label
    if analyze(net, inputs, eps, true_label, dataset):
        print('verified')
    else:
        print('not verified')
    


if __name__ == '__main__':
    main()

