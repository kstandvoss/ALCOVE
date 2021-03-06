local Association, Parent = torch.class('nn.Association', 'nn.Module')

--[[
Torch Neural Network Module to add Attention Layer
@size dimension of input Tensor
]]--
function Association:__init(n_output, n_exemplars)
    Parent.__init(self)
    self.n_output = n_output
    self.n_exemplars = n_exemplars
    self.weights = torch.Tensor(n_output, n_exemplars):zero()
    self.grad = torch.Tensor(#self.weights):zero()
end

--@input Tensor with first dimension of size @size 
function Association:updateOutput(input)
    output = torch.Tensor(self.n_output)
    for k = 1,self.n_output do
        output[k] = torch.sum(torch.cmul(self.weights[k],input))
    end
    self.output = output
    return self.output
end
 


--@input Tensor with input of Backward-call of previous module
--@gradOutput Gradient output of previous module
function Association:updateGradInput(input, gradOutput)
    gradInput = torch.Tensor(#self.weights)
    gradInput = gradOutput:resize(1,self.n_output) * self.weights
    self.gradInput = gradInput
    return self.gradInput
end

function Association:updateParameters(learningRate)
    self.weights:add(learningRate, self.grad)
end        


function Association:zeroGradParameters()
    self.grad = self.grad:zero()
end

--@input Tensor with input of Backward-call of previous module
--@gradOutput Gradient output of previous module 
--@scale Factor for parameter accumulation
function Association:accGradParameters(input, gradOutput, scale)  
   gradient = torch.ger(gradOutput:resize(2),input:resize(8)) 
   self.grad =  self.grad + torch.mul(gradient,scale) 
end