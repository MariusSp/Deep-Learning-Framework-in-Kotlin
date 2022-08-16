package functions

import structure.Tensor

fun batchGradientDecent(learningRate: Float, derweights: ArrayList<Tensor>, weights: Tensor) {
    if (derweights.first().elements.size > 10000) {
        derweights.parallelStream().parallel().forEach { i ->
            for (j in 0 until i.shape.volume) {
                weights.elements[j] -= i.elements[j] * learningRate
            }
        }
    } else {
        derweights.forEach { i ->
            for (j in 0 until i.shape.volume) {
                weights.elements[j] -= i.elements[j] * learningRate
            }
        }
    }
}
