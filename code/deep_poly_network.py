import torch
from networks import Normalization
from resnet import BasicBlock, ResNet
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DP_object:
    """
    This is the abstract object we will push through our Deep Poly Network. Because we transform every layer to a linear transform,
    we Flatten and normalize in the beginning of this object, then send this through our DP linear, conv and ReLU layers.
    Main attributes
    self.rel_bounds:    List of relative constraints in matrix form, how a layer depends on it's parent layer. 
                        Matrix has form (output x inputs + 1). Output is nr. of features in layer, input is from previous layer and 1 bias column.
    self.abs_bounds:    In each layer we compute the exact lower and upper bounds of this layer through backsubstitution of rel_bounds.
                        This is important to get best possible bounds for ReLU layers.
                    
    """
    def __init__(self, inputs, eps, dataset) -> None:
        lb = inputs - eps
        ub = inputs + eps
        lb = lb.clamp(0,1)
        ub = ub.clamp(0,1)

        #Every network is normalized first. 
        lb = Normalization(device="cpu", dataset=dataset).forward(lb)
        ub = Normalization(device="cpu", dataset=dataset).forward(ub)

        #We will recode all layers so that they become simple matrix multiplications. With that we can easily back-substitute our constraints.
        #This means we only work with flattened tensors. We also only need to substitute back to this layer, as normalization does nothing really.
        lb = nn.Flatten().forward(lb) #!nn.Flatten will keep batch dimension. Thus our input will become (1, cat(all other dimensions))
        ub = nn.Flatten().forward(ub)


        #Will store (lb, ub) tuples in these list for every step we take in the layers
        self.rel_bounds = [(lb.reshape(-1,1), ub.reshape(-1,1))] #For backsubstitution expects previous nodes in columns, need to reshape
        self.abs_bounds = [(lb.reshape(-1,1), ub.reshape(-1,1))] #For input layer

    def append_relbound(self, rel_bound):
        """ 
        In each layer we need to append the rel_bounds to our list, in order to get correct backsubstitution later.
        """
        self.rel_bounds.append(rel_bound)

    def backsub(self):
        """
        This function taks as input the relative bounds and substitutes back to get the best possible absolute lb and ub for this layer.T_destination.
        We do this by simple matrix, matrix multiplication, as we have ensured all our rel_bounds are in the correct format.
        The format of rel_bounds should be (out_features, in_features + bias_column). Every row describes how one node depends on the nodes
        of the previous layer. 
        Remember that input (lb, ub) are first tuple of rel_bounds, so for exact bounds can substitute all the way through list. 
        """

        rel_bounds = self.rel_bounds
        #If we don't have relative bounds yet (so just inputs) return the input bounds
        if len(rel_bounds) == 1:
            return rel_bounds[0]
        
        #Initializing back-substitution with current bounds
        current_lb, current_ub = rel_bounds[-1]
        
        #If we have a relative constraint, can back-substitute. Loop backward through relative constraint with linear matrix multiplication.
        for i in range(len(rel_bounds)-1, 0, -1):

            prev_lb, prev_ub = rel_bounds[i - 1]

            #Appending a row of 0 and 1 at end of prev bounds for bias term of current layer
            bias_row = torch.zeros(size = (1, prev_lb.shape[-1])) #Needs to be of shape of columns
            bias_row[0, -1] = 1
            prev_lb = torch.cat((prev_lb, bias_row), dim = 0)
            prev_ub = torch.cat((prev_ub, bias_row), dim = 0)
            
            #For lower bound: 
            weight_pos = F.relu(current_lb)
            weight_neg = F.relu(-current_lb)
            new_lb = torch.matmul(weight_pos, prev_lb) - torch.matmul(weight_neg, prev_ub)

            #For upper bound choose upper bound constraint
            weight_pos = F.relu(current_ub)
            weight_neg = F.relu(-current_ub)
            new_ub = torch.matmul(weight_pos, prev_ub) - torch.matmul(weight_neg, prev_lb)

            #Update the substituted relative bounds
            current_lb = new_lb
            current_ub = new_ub
         
        self.abs_bounds.append((current_lb, current_ub))


class DeepPLinear(nn.Module):
    def __init__(self, layer: nn.Linear) -> None:
        super().__init__()
        #Needs weights of layer
        self.weight = layer.weight.detach()
        self.bias = layer.bias.detach()
        self.out_features = layer.out_features
        """
        Returns a tuple (rel_lb, rel_ub) for the relative constraint matrix of this layer.
        #In each row is the relative constraint of one node in the current layer wrt. to nodes (and bias) of previous layer.
        #Each column represents how a node/variable of previous layer influences the nodes in current layer.
        #Relative constraint is of dimension output x input + 1. Output: current nodes, input: previous nodes, +1 for bias
        """
        #Weights will be (out_features x in_features), bias will be (,out_features), which means need to unsqueeze bias to work with cat
        constr = torch.cat((self.weight, self.bias.unsqueeze(-1)), dim = 1)
        #For linear layer rel. upper and lower bound are same
        self.rel_bound = (constr, constr)
    
    def forward(self, x: DP_object):

        #Need to update our relative bounds in x
        x.append_relbound(self.rel_bound)

        return x

class DeepPReLU(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = self.in_features
        self.register_parameter("alphas", None)
        #self.alphas = nn.parameter.Parameter(torch.ones(self.in_features))

    def forward(self, x: DP_object):
        #Only backward in ReLU layers to get exact abs bounds:
        x.backsub()
        self.prev_lb, self.prev_ub = x.abs_bounds[-1]

        #Now work with computed exact abs bounds
        prev_lb, prev_ub =self.prev_lb, self.prev_ub
        prev_lb.detach()
        prev_ub.detach()

        #Initializing alphas with area heuristic in first forward, if abs(lb) > abs(ub) then alpha = 0, otherwise 1
        if self.alphas is None: 
            heuristic = torch.where(torch.abs(prev_lb.squeeze()) > torch.abs(prev_ub.squeeze()), 0.0, 1.0)
            self.alphas = nn.Parameter(heuristic)

        #Lower bound

        alpha = torch.clip(self.alphas,0,1)
        alpha = torch.where(prev_lb.squeeze() > 0.0, torch.tensor(1, dtype=alpha.dtype), alpha) #Where lb greater 1, entry corresponds to identity == 1, else it's alpha
        alpha = torch.where(prev_ub.squeeze() <= 0.0, torch.tensor(0, dtype=alpha.dtype), alpha) #Where ub smaller 1, entry is 0, else it's alpha

        #Lower bound constraints with added column for bias
        current_lb = torch.cat((torch.diag(alpha), torch.zeros(alpha.shape[0], 1)), dim=1) 
        
        #Upper bound
        slope_ub = prev_ub/(prev_ub-prev_lb + torch.finfo(prev_lb.dtype).eps) #Slope for upper bound constraint
        slope_ub = torch.where(prev_lb > 0.0, torch.tensor(1, dtype = slope_ub.dtype), slope_ub)
        slope_ub = torch.where(prev_ub < 0.0, torch.tensor(0, dtype = slope_ub.dtype), slope_ub)

        bias_ub = -prev_lb*slope_ub #Bias for upper bound constraint
        bias_ub = torch.where((prev_lb > 0.0) | (prev_ub < 0.0), torch.tensor(0, dtype=bias_ub.dtype), bias_ub) #no bias if we can be exact

        current_ub = torch.cat((torch.diag(slope_ub.squeeze()), bias_ub), dim = 1) #Upper constraint matrix

        #rel_bound
        self.rel_bound = (current_lb, current_ub)

        x.append_relbound(self.rel_bound)

        return x



class DeepPConv(nn.Module):
    def __init__(self, layer: nn.Conv2d, in_features: int):
        """
        Initialization of our DeepP-Convolutional layer:
        layer: it is the Conv2d module from the original network
        in_features: total amount of input neurons from original network (Tensor X x Y x Z -> X*Y*Z input neurons)
        """
        super().__init__()
        #At first, we copy all the attributes from the original network layer:
        self.weight = layer.weight.detach()
        self.padding = layer.padding[0]
        self.stride = layer.stride[0]
        self.kernel_size = layer.kernel_size[0]
        self.in_features = in_features
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.in_height = int(np.sqrt(self.in_features / self.in_channels))
        self.in_width = int(np.sqrt(self.in_features / self.in_channels))
        self.out_height = int((self.in_height + 2 * self.padding - self.kernel_size) / self.stride + 1)
        self.out_width = int((self.in_width + 2 * self.padding - self.kernel_size) / self.stride + 1)
        self.out_features = self.out_channels * self.out_height * self.out_width

        #If the original network does not have a bias, we will add column of zeros
        if layer.bias == None:
            self.bias = torch.zeros(layer.out_channels)
        else:
            self.bias = layer.bias.detach()

        #For normalization purposes, we keep the original layer as an attribute
        self.conv = layer.requires_grad_(False)
        #After copying all attributes, we build a 2D Matrix from the kernel weights
        self.matrix = self.conv2d_matrix()
        self.rel_bound = (self.matrix, self.matrix)
    
    def conv2d_matrix(self):
        # Intuition: treat padding as normal input dimensions and remove it in the end.
        in_width_pad = self.in_width + 2 * self.padding
        in_height_pad = self.in_height + 2 * self.padding
        in_spatial_dim_pad= in_height_pad * in_width_pad
        # Matrix storing results 
        results_mat = torch.zeros((self.out_height * self.out_width * self.out_channels,
                           in_spatial_dim_pad * self.in_channels))
        # Construct fillers for the rows
        row_fillers_length = (self.in_channels - 1) * in_spatial_dim_pad +\
                          (self.kernel_size - 1) * in_width_pad + self.kernel_size
        fillers_row = torch.zeros((self.out_channels, row_fillers_length))
        for output_ch_id in range(self.out_channels):
            for input_ch_id in range(self.in_channels):
                for kernel_row_id in range(self.kernel_size):
                    start_id = input_ch_id * in_spatial_dim_pad + kernel_row_id * in_width_pad
                    fillers_row[output_ch_id, start_id: start_id + self.kernel_size] =\
                        self.weight[output_ch_id, input_ch_id, kernel_row_id]

        # Now we fill the rows of the result matrix using the previously built row-fillers
        for output_ch_id in range(self.out_channels):
            for output_height_id in range(self.out_height):
                for output_width_id in range(self.out_width):
                    row_offset = output_height_id * self.stride * in_width_pad + output_width_id * self.stride
                    out_neuron_id = output_ch_id * self.out_height * self.out_width +\
                                     output_height_id * self.out_width + output_width_id
                    results_mat[out_neuron_id, row_offset: row_offset + row_fillers_length] = fillers_row[output_ch_id]

        # Now that the result matrix is filled, we remove the padding to complete the Convolution:
        duplicates = []
        for input_ch_id in range(self.in_channels):
            for input_height_id in range(in_height_pad):
                for input_width_id in range(in_width_pad):
                    if input_width_id < self.padding or input_width_id >= self.padding + self.in_width:
                        duplicates.append(
                            input_ch_id * in_spatial_dim_pad + input_height_id * in_width_pad + input_width_id)
                if input_height_id < self.padding or input_height_id >= self.padding + self.in_height:
                    start_id = input_ch_id * in_spatial_dim_pad + input_height_id * in_width_pad
                    duplicates = duplicates + list(range(start_id, start_id + in_width_pad))

        # cut away the duplicate columns:
        duplicates = list(np.unique(np.array(duplicates)))
        results_mat = torch.from_numpy(np.delete(results_mat.numpy(), duplicates, axis=1))

        # add bias column
        results_mat = torch.cat([results_mat, torch.repeat_interleave(self.bias, self.out_width * self.out_height).unsqueeze(1)], dim=1)
        self.weight = results_mat[:,:-1]
        self.bias = results_mat[:,-1]
        return results_mat

    def forward(self, x: DP_object):
        x.append_relbound(self.rel_bound)
        return x


class DeepIdentity(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = self.in_features
        self.matrix = torch.cat((torch.eye(self.in_features), torch.zeros(self.in_features,1)), dim = 1)
        self.rel_bound = (self.matrix, self.matrix)
    
    def forward(self, x: DP_object):
        x.append_relbound(self.rel_bound)
        return x
        

class DeepPadBlock(nn.Module):
    """ 
    Because we are treating resnet block as fully connected, need this befor to "double" the input for block, and after to add output of block
    together. Otherwise matrix multiplication won't work. 
    """
    def __init__(self, in_features: int, before_block: bool) -> None:
        super().__init__()
        self.in_features = in_features
        if before_block:
            #Need to stack weights in dim 0
            self.out_features = 2*self.in_features
            self.weights = torch.cat((torch.eye(self.in_features), torch.eye(self.in_features)), dim = 0)
            self.bias = torch.zeros((self.out_features,1))
        else:
            #Need to stack weights in dim 1
            self.out_features = int(0.5*self.in_features)
            self.weights = torch.cat((torch.eye(self.out_features), torch.eye(self.out_features)), dim = 1)
            self.bias = torch.zeros((self.out_features,1))

        self.matrix = torch.cat((self.weights, self.bias), dim = 1)
        self.rel_bound = (self.matrix, self.matrix)
    
    def forward(self, x:DP_object):
        x.append_relbound(self.rel_bound)
        return x


class DeepBlock(nn.Module):
    """ 
    We're treating a Block as a fully connected layer. For this we first treat the whole network ending in path a as a fully sequential network
    and the whole network ending in path b as another fully connected sequential network (so that we can get exact matrices and bounds for each
    path). We then stack the rel_bound matrices of path a and path b diagonally and double the input (stack input nodes twice horizontally) of
    the block with a matrix multiplication. Like this we can then go through the whole block as a fully connected sequential layer. At the end
    of the block we get a vector that is compromised of the output of path a stacked on top of the output of path b. With a matrix multiplication
    (transpose of doubling in beginning) we again add the corresponding outputs together.

    in_features (int): number of in_features for the blocks (is half of what the stacked matrices will multiply with)
    """
    def __init__(self, layer: BasicBlock, in_features) -> None:
        super().__init__()
        #To-Do:
        self.in_features = in_features

        #Initiating Paths
        self.path_a = layer.path_a.requires_grad_(False)
        self.path_b = layer.path_b.requires_grad_(False)
        self.path_a_layers = []
        self.path_b_layers = []

        #Creating Layers list for paths
        self.block_builder(self.path_a, self.path_a_layers)
        self.block_builder(self.path_b, self.path_b_layers)    
        self.path_a_length = len(self.path_a_layers)
        self.path_b_length = len(self.path_b_layers)

        #If a Block is shorter than the other, it has to be padded by identity layers:
        self.block_length = max(self.path_a_length, self.path_b_length)  
        self.padder(self.path_a_layers, self.block_length)
        self.padder(self.path_b_layers, self.block_length)

        #Out features of block is the last layers out features (picked block a here, but they should have same dimension so no difference)
        assert self.path_a_layers[-1].out_features == self.path_b_layers[-1].out_features, "Blocks don't have same output dimension!"
        self.out_features = self.path_a_layers[-1].out_features

        #Creating Networks:
        self.path_a_net = nn.Sequential(*self.path_a_layers)
        self.path_b_net = nn.Sequential(*self.path_b_layers)
    
    def forward(self, x:DP_object):
        """ 
        First create individual matrices for paths. Then stack them together diagonally. Append a diagonally stacked matrix for each layer
        in the longer path.
        """
        # Need to create matrices for each path, so that we can stack them later. Thus copy x and pretend whole net is sequential and ending
        # in specified path.

        #Path a
        self.path_a_net(x)
        list_a = x.rel_bounds[-self.block_length:]
        x.rel_bounds = x.rel_bounds[:-self.block_length]

        #Path b
        self.path_b_net(x)
        list_b = x.rel_bounds[-self.block_length:]
        x.rel_bounds = x.rel_bounds[:-self.block_length]

        #Appending matrices to rel_bounds of x
        #Block padder before
        x.append_relbound(self.block_padder(self.in_features, before_block=True))

        #Stacking matrices diagonally
        for i in range(self.block_length):
            lb_a, ub_a = list_a[i]
            lb_b, ub_b = list_b[i]

            #Lower bound
            #Extract weights and bias:
            lb_a_bias = lb_a[:,-1]
            lb_a_weights = lb_a[:,:-1]
            lb_b_bias = lb_b[:, -1]
            lb_b_weights = lb_b[:, :-1]

            #Stack weights diagonally and biases on top of each other, yielding again a matrix of (output X input+1)
            lb_weights = torch.block_diag(lb_a_weights, lb_b_weights)
            lb_bias = torch.cat((lb_a_bias, lb_b_bias), dim = 0).reshape(-1,1)
            lb = torch.cat((lb_weights, lb_bias), dim = 1)

            #Upper bouond
            #Extract weights and bias:
            ub_a_bias = ub_a[:,-1]
            ub_a_weights = ub_a[:,:-1]
            ub_b_bias = ub_b[:, -1]
            ub_b_weights = ub_b[:, :-1]

            #Stack weights diagonally and biases on top of each other, yielding again a matrix of (output X input+1)
            ub_weights = torch.block_diag(ub_a_weights, ub_b_weights)
            ub_bias = torch.cat((ub_a_bias, ub_b_bias), dim = 0).reshape(-1,1)
            ub = torch.cat((ub_weights, ub_bias), dim = 1)

            x.append_relbound((lb,ub))

        #Adding block padding layer at end of block
        # Block output is outputs of both paths added up. For the adding padder we thus have self.out_features*2 features. 
        x.append_relbound(self.block_padder(in_features=2*self.out_features, before_block=False)) 

        return x
        
    def padder(self, path_list:list, max_length):
        """ 
        This function pads a list of layer modules with Identity Modules to a certain length
        """
        while len(path_list) < max_length:
            path_list.append(DeepIdentity(path_list[-1].out_features))
    
    def block_builder(self, path: nn.Sequential, path_layers: list):
        """ 
        Function builds a list of DP Modules, based on the architecture of the passed Sequential network. List can then be passed to nn.Sequential()
        path: Sequential (Block) Network to be converted to a DP Network
        """
        in_features = self.in_features
        for layer in path.children():
            appender(layer, path_layers, in_features)
            in_features = path_layers[-1].out_features
    
    def block_padder(self, in_features, before_block):
        """
        Returns relative bound matrices for block paddings. Matrices will be stacked identity matrices with a bias column as last column.
        If before block they will be stacked in dim 0 (doubling input), if after block in dim 1 (adding block outputs). 
        """
        if before_block:
            #Need to stack weights in dim 0
            out_features = 2*in_features
            weights = torch.cat((torch.eye(in_features), torch.eye(in_features)), dim = 0)
            bias = torch.zeros((out_features,1))
        else:
            #Need to stack weights in dim 1
            out_features = int(0.5*in_features)
            weights = torch.cat((torch.eye(out_features), torch.eye(out_features)), dim = 1)
            bias = torch.zeros((out_features,1))

        matrix = torch.cat((weights, bias), dim = 1)
        rel_bound = (matrix, matrix)
        return rel_bound


class DeepPOverlap(nn.Module):
    """
    Will return the computed lower and upper bounds, as well as the overlap between true class and other.
    We want this overlap/distance to be positive for all entries. 
    """
    def __init__(self, in_features, true_label) -> None:
        super().__init__()
        self.true_label = true_label
        self.in_features = in_features
        self.out_features = self.in_features-1

        #Constructing a matrix that subtracts true class from all other classes. If all overlap negative then verified
        self.bias = torch.zeros(self.out_features, 1)
        iden = torch.eye(self.out_features)
        self.weight = torch.cat((iden[:,:self.true_label], -torch.ones(self.out_features, 1), iden[:,self.true_label:]), dim = 1)
        self.matrix = torch.cat((self.weight, self.bias), dim = 1)
        self.rel_bound = (self.matrix, self.matrix)

    
    def forward(self, x: DP_object):
        #Append relative bounds
        x.append_relbound(self.rel_bound)

        #Compute abs bounds:
        x.backsub()
        lb, ub = x.abs_bounds[-1]

        return (lb.reshape(1,-1), ub.reshape(1,-1))


##################
#Helper Functions#
##################

def appender(layer, layers:list, in_features: int):
        """ 
        Function checks for type of layer and adds corresponding DeepPoly Module to layers list. Used for network builder.
        layer: layer of original net that needs to be replaced by its DeepPoly Alternative in DeepPoly Net. 
        layers: list containing the previous DeepPoly Layers. Will later be used to construct a sequential network. 
        in_features: the number of in features for this layer. Equals the out features of previous layer.

        returns: layers list appended with the new DeepPoly Layer.
        """
        if type(layer) == nn.Linear:
                layers.append(DeepPLinear(layer))
        if type(layer) == nn.ReLU:
            layers.append(DeepPReLU(in_features))
        if type(layer) == nn.Conv2d:
            layers.append(DeepPConv(layer, in_features))
        if type(layer) == nn.BatchNorm2d:
            #If Batchnormalization need to normalize last convolution
            conv = merge_conv_and_bn(layers[-1].conv, layer)
            layers[-1] = DeepPConv(conv, layers[-1].in_features)
        if type(layer) == BasicBlock:
            layers.append(DeepBlock(layer, in_features))
        if type(layer) == nn.Identity:
            layers.append(DeepIdentity(in_features))



def merge_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d): 
    """ 
    Function fuses together a conv layer with it's subsequent batch normalization layer. The updated layer is then fed to the DeepPConv constructor.
    This way we did not have to change our DeepPConv Class.
    Source: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/ (02.12.2022)
    conv: Convolution Layer (unnormalized)
    bn: Batch Normalization Layer

    returns: a normalized nn.Conv2d layer.
    """
    # initialisation of Convolution and Batchnorm Merge
    mergedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    
    # Preparation of filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))

    mergedconv.weight.detach().copy_( torch.mm(w_bn, w_conv).view(mergedconv.weight.size()) )
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0) )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    mergedconv.bias.detach().copy_( torch.matmul(w_bn, b_conv) + b_bn )
    # we're done
    return mergedconv



def network_builder(net, inputs, true_label, error_check = False):
    """ 
    This function builds a DeepPoly Network with abstract transformations as layers, from the original input net.
    net: input net to be rebuilt for verification.
    inputs: point to be verified.
    true_label: true label of point

    returns: A DeepPoly Network of type nn.Sequential()
    """
    #Building Network layer for layer

    layers = []
    in_features = inputs.reshape(-1).shape[0]
    for layer in net.layers:
        appender(layer, layers, in_features)
        if len(layers) == 0:
            in_features = in_features
        else: 
            in_features = layers[-1].out_features
            
    #Error check or not
    if not error_check:
        layers.append(DeepPOverlap(in_features, true_label))
    DP_net = nn.Sequential(*layers)
    return DP_net
    

def resnet_builder(net, inputs, true_label, error_check = False):
    """ 
    Function builds a DeepPoly Implementation based on an input net. Differs from network builder, as resnets are built differently.
    net: input ResNet to be rebuilt for verification.
    inputs: point to be verified.
    true_label: true label of point

    returns: A DeepPoly Network of type nn.Sequential()
    """

    layers = []
    in_features = inputs.reshape(-1).shape[0]

    #Extract Resnet portion
    for module in net.modules():
        if type(module) == ResNet:
            resnet = module
    
    #Go through resnet layers
    for child in resnet.children():

        #If child is a sequential layer (used for blocks), need to unpack it further
        if type(child) is nn.Sequential:
            for grandchild in child.children():
                appender(grandchild, layers, in_features)
                in_features = layers[-1].out_features
        
        #If it is just another layer, add it
        else:
            appender(child, layers, in_features)
    
        #New in_features for next layer
        if len(layers) == 0:
            in_features = in_features
        else: 
            in_features = layers[-1].out_features

    #Error check or not
    if not error_check:
        layers.append(DeepPOverlap(in_features, true_label))
    DP_net = nn.Sequential(*layers)
    return DP_net

