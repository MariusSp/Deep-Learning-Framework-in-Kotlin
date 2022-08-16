package inputs

import structure.Tensor

interface Input {
    val input: List<Tensor>
    val labels: List<Tensor>

}