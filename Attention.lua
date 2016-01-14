local Attention, Parent = torch.class('nn.Attention', 'nn.Module')

--[[
Torch Neural Network Module to add Attention Layer
@size dimension of input Tensor
]]--
function Attention:__init(n_input, n_exemplars, exemplars, specificity, metric, similarity)
    Parent.__init(self)
    self.n_input = n_input
    self.n_exemplars = n_exemplars
    self.exemplars = exemplars
    self.specificity = specificity
    self.metric = metric
    self.similarity = similarity 
    self.alphas = torch.Tensor(n_input):fill(1/self.n_input)
    self.grad = torch.Tensor(#self.alphas):zero()
end

--@input Tensor with first dimension of size @size 
function Attention:updateOutput(input)
    output = torch.Tensor(self.n_exemplars)
    for j = 1,self.n_exemplars do
        output[j] = torch.sum(torch.cmul(self.alphas, torch.pow(torch.abs(self.exemplars[j] - input), self.metric)))
    end           
    output = torch.mul(torch.pow(output, (self.similarity/self.metric)), -self.specificity)
    output = output:exp()
    self.output = output
    return self.output
end



--@input Tensor with input of Backward-call of previous module
--@gradOutput Gradient output of previous module
function Attention:updateGradInput(input, gradOutput)
    self.gradInput = input
    return self.gradInput
end

function Attention:updateParameters(learningRate)
    self.alphas:add(-learningRate, self.grad)
    self.alphas[torch.lt(self.alphas,0)] = 0
end

function Attention:zeroGradParameters()
    self.grad = self.grad:zero()
end

--@input Tensor with input of Backward-call of previous module
--@gradOutput Gradient output of previous module 
--@scale Factor for parameter accumulation
function Attention:accGradParameters(input, gradOutput, scale)
   distance = torch.Tensor(self.n_input)
   for i = 1,self.n_input do
        sum = 0
        for j = 1,self.n_exemplars do 
            sum = sum + (self.specificity * gradOutput[1][j] * self.output[j] * torch.abs(self.exemplars[j][i] - input[i]))
        end
        distance[i] = sum    
   end
   self.grad = self.grad + distance --torch.mul(distance, scale)  
end