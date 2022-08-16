package layers

import structure.Tensor

interface Layer {
    fun forward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>)

    fun backward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>, forwardInTensors: ArrayList<Tensor>)
}