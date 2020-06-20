import os
import cv2
import numpy as np
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
from torch.nn import functional as F


# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='episode'):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
    ys = np.asarray(ys_population, dtype=np.float32)
    ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
    trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
    data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
  else:
    data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
  plotly.offline.plot({
    'data': data,
    'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)


def write_video(frames, title, path=''):
  frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
  _, H, W, _ = frames.shape
  writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
  for frame in frames:
    writer.write(frame)
  writer.release()

def imagine_ahead(prev_state, prev_belief, policy, transition_model, planning_horizon=12):
  '''
  imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
  Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200]) 
  Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
          torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
  '''
  flatten = lambda x: x.view([-1]+list(x.size()[2:]))
  prev_belief = flatten(prev_belief)
  prev_state = flatten(prev_state)
  
  # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
  T = planning_horizon
  beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
  beliefs[0], prior_states[0] = prev_belief, prev_state

  # flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
  # start = {k: flatten(v) for k, v in post.items()}


  # Loop over time sequence
  for t in range(T - 1):
    _state = prior_states[t]
    # print(beliefs[t], _state)
    actions = policy.get_action(beliefs[t],_state)
    # Compute belief (deterministic hidden state)
    # act_fn_inp = torch.cat([_state, actions], dim=1)
    # print("imagine", act_fn_inp.size())
    hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
    beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
    # Compute state prior by applying transition dynamics
    hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
    prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
    prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.min_std_dev
    prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
  # Return new hidden states
  # imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
  imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
  return imagined_traj

def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  # print("input of lambda return:",imged_reward.size(), value_pred.size(), bootstrap.size(), lambda_)
  # (11,50), (11,50), (50), 0.95
  next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
  # print("next value", next_values.size())
  discount_tensor = discount * torch.ones_like(imged_reward) #pcont
  inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
  last = bootstrap
  indices = reversed(range(len(inputs)))
  outputs = []
  for index in indices:
    inp, disc = inputs[index], discount_tensor[index]
    last = inp + disc*lambda_*last
    outputs.append(last)
    # print("last:", last.size())
  outputs = list(reversed(outputs))
  outputs = torch.stack(outputs, 0)
  # outputs = [torch.stack(,0) for x in outputs]
  # print("stacked output: ",outputs.size())
  # print("returns:", returns)
  returns = outputs
  return returns