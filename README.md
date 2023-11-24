# A neural-network reinforcement-learning model of domestic chicks that learn to localize the centre of closed arenas

The code is a neural network implementation of the TD(0) reinforcement learning actor-critic architecture. The model is based on the work in [Mannella, F., & Baldassarre, G. (2006)](https://royalsocietypublishing.org/doi/10.1098/rstb.2006.1966), which aims to reproduce and understand the behaviors of domestic chicks in experiments where they are trained to locate food in different-sized arenas. The model provides insights into the chicks' navigation behaviors and suggests possible cognitive mechanisms underlying their actions. 

In the simulator, the agent sees a clear 3D view of the arena walls' edges. The neural architecture is a perceptron equipped with two sets of action units—one dedicated to speed and the other to rotation—alongside an evaluation unit.The learned weights help us understand how the agent manages its actions in different visual situations.

<table>
<tbody>
  <tr>
    <td><img  src="docs/demo.gif" width="300">
    <td><img src="docs/model.png" width="510">
  </tr>
  <tr>
    <td  colspan="2"><img width="840" src="docs/analysis.png"> </td>
  </tr>
</tbody>

