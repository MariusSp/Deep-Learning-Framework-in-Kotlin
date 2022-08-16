package inputs

import structure.Tensor

class InputSimple(override val input: List<Tensor>, override val labels: List<Tensor>) : Input