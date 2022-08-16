package layers

import structure.Tensor

class Activation(private var actfk: (x: Float) -> Float, private var actfkder: (x: Float) -> Float) : Layer {
    private fun activationFunction(inTensor: Tensor, outTensor: Tensor) {
        outTensor.elements = inTensor.elements.map { actfk(it) }.toFloatArray()
    }

    private fun deractivationFunction(inTensor: Tensor, outTensor: Tensor, forwardInTensor: Tensor) {
        forwardInTensor.elements = forwardInTensor.elements.map(actfkder).toFloatArray()
        for (i in 0 until forwardInTensor.shape.axis[0]) {
            outTensor.elements[i] = inTensor.elements[i] * forwardInTensor.elements[i]
        }
    }

    override fun forward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>) {
        if (inTensors.size != outTensors.size) {
            throw Exception("list size mismatch")
        }
        inTensors.indices.toSet().parallelStream().parallel().forEach { activationFunction(inTensors[it], outTensors[it]) }
    }

    override fun backward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>, forwardInTensors: ArrayList<Tensor>) {
        inTensors.indices.toSet().parallelStream().parallel().forEach { deractivationFunction(inTensors[it], outTensors[it], forwardInTensors[it]) }
    }
}