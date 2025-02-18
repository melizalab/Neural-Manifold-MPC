import torch
import torch.nn as nn
import snntorch as snn

class Net(nn.Module):
    '''
    Creates a three layer SNN where each layer is recurrent with itself:
    - Layer 1 (Sensory): receives direct stimulus input
    - Layer 2 (Reservoir): all-to-all connections with the sensory layer
    - Layer 3 (Output): gets inputs from reservoir and neuron with highest activation is predicted digit

    The linear functions (learnable weights) in network are referred to as 'dendrites' since they act as a basic model
    of the dendritic tree.
    '''
    def __init__(self,params):
        self.sensory_layer_size = params['sensory_layer_size']
        self.reservoir_layer_size = params['reservoir_layer_size']
        self.stimulus_n_steps = params['stimulus_n_steps']
        self.is_online = params['is_online']
        super().__init__()

        # Init Neurons
        self.sensory_neurons = snn.RLeaky(beta=self.beta,reset_mechanism='zero',threshold = 1,linear_features=self.sensory_layer_size)
        self.reservoir_neurons = snn.RLeaky(beta=self.beta,reset_mechanism='zero',threshold=1,linear_features=self.reservoir_layer_size)
        self.output_neurons = snn.RLeaky(beta=self.beta,reset_mechanism='zero',threshold=1,linear_features=10)

        # Hard-coded values for MNIST problem
        stimulus_size = 28*28
        self.output_layer_size = 10

        ### INITIALIZE WEIGHTS (named dendrites because I'm in a neuro lab and I can)###
        # Layer 1 (Dendrites)
        self.sensory_dendrites = nn.Linear(stimulus_size,self.sensory_layer_size)
        # Layer 2 (Reservoir)
        self.reservoir_dendrites = nn.Linear(self.sensory_layer_size,self.reservoir_layer_size)
        # Layer 3 (Output)
        self.output_dendrites = nn.Linear(self.reservoir_layer_size,self.output_layer_size)

        if self.is_online == True:
            # The membrane voltages are initialized when the class is first called.
            # This allows the voltages to accumulate across time steps when the stimulus
            # is presented in an online fashion.
            self.initialize_spikes_and_voltages()

    def initialize_spikes_and_voltages(self):
        sensory_spikes, sensory_V = self.sensory_neurons.init_rleaky()
        reservoir_spikes, reservoir_V = self.reservoir_neurons.init_rleaky()
        output_spikes, output_V = self.output_neurons.init_rleaky()

        # Register buffers for device consistency
        self.register_buffer('sensory_spikes', sensory_spikes)
        self.register_buffer('sensory_V', sensory_V)
        self.register_buffer('reservoir_spikes', reservoir_spikes)
        self.register_buffer('reservoir_V', reservoir_V)
        self.register_buffer('output_spikes', output_spikes)
        self.register_buffer('output_V', output_V)


    def forward(self,stimulus):
        '''
        Structure of a single trial:
        The SNN simulates the response to a sequence of MNIST digits. Each time the forward pass is called, this
        is simulating the response of the network to one trial. This is why the neurons are initialized at the beginning
        of each forward pass (see below). This resets the membrane voltage V (e.g. sensory_V for sensory neurons).
        The sequence is stimulus_n_steps time steps long and the input into the sensory layer at each t_step is the value
        of the stimulus at that time step (e.g. t_step = 3 -> input = stimulus(3)).
        '''
        if self.is_online == False:
            # Reset membrane voltages of neurons at beginning of each trial.
            self.initialize_spikes_and_voltages()

        # Set up empty arrays to record spikes from neurons.
        sensory_spike_rec = []
        reservoir_spike_rec = []
        output_spike_rec = []
        output_V_trace = []

        # Present stimulus to network for stimulus_n_steps time points
        for t_step in range(self.stimulus_n_steps):
            # Sensory Layer
            sensory_input = self.sensory_dendrites(stimulus)
            sensory_noise = self.noise_scaling * torch.randn(len(stimulus), self.sensory_layer_size,device=sensory_input.device)
            sensory_input_total = sensory_input+sensory_noise
            self.sensory_spikes,self.sensory_V = self.sensory_neurons(sensory_input_total,self.sensory_spikes,self.sensory_V)
            if type(self.clamp_value) is not str:
                self.sensory_V = torch.clamp(self.sensory_V, min=self.clamp_value)
            sensory_spike_rec.append(self.sensory_spikes)

            # Reservoir Layer
            reservoir_input = self.reservoir_dendrites(self.sensory_spikes)
            reservoir_noise = self.noise_scaling*torch.randn(len(stimulus),self.reservoir_layer_size,device=sensory_input.device)
            reservoir_input_total = reservoir_input+reservoir_noise
            self.reservoir_spikes,self.reservoir_V = self.reservoir_neurons(reservoir_input_total,self.reservoir_spikes,self.reservoir_V)
            if type(self.clamp_value) is not str:
                self.reservoir_V = torch.clamp(self.reservoir_V, min=self.clamp_value)
            reservoir_spike_rec.append(self.reservoir_spikes)

            # Output Layer
            output_input = self.output_dendrites(self.reservoir_spikes)
            output_noise = self.noise_scaling*torch.randn(len(stimulus),self.output_layer_size,device=sensory_input.device)
            output_input_total = output_input+output_noise
            self.output_spikes,self.output_V = self.output_neurons(output_input_total,self.output_spikes,self.output_V)
            if type(self.clamp_value) is not str:
                self.output_V = torch.clamp(self.output_V, min=self.clamp_value)
            output_spike_rec.append(self.output_spikes)

            if self.return_out_V == True:
                output_V_trace.append(self.output_V)

        # Stack spike train recordings for convenience
        sensory_spike_rec = torch.stack(sensory_spike_rec)
        reservoir_spike_rec = torch.stack(reservoir_spike_rec)
        output_spike_rec = torch.stack(output_spike_rec)
        if self.return_out_V == True:
            output_V_trace = torch.stack(output_V_trace)

        return sensory_spike_rec,reservoir_spike_rec,output_spike_rec,output_V_trace

class rLIF(Net):
    """
    This architecture is made up of LIF neurons with no synaptic dynamics but recurrence.
    """
    def __init__(self,params,return_out_V = True):
        self.beta = params['beta']
        self.noise_scaling = params['noise_scaling']
        self.batch_size = params['batch_size']
        self.clamp_value = params['clamp_value']
        """
        parameters in params dictionary -->
            beta: decay rate of membrane voltage
            sensory_layer_size: number of neurons in the input layer
            reservoir_layer_size: number of neurons in hidden layer
            stimulus_n_steps: how many time steps image is 'shown' to network
        """
        super().__init__(params)
        self.return_out_V = return_out_V

