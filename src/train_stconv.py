#!/usr/bin/env python3
"""Ponto de entrada estável do trainer STConv configurável.

Mantém compatibilidade com o harness experimental originalmente criado como
``train_stconv_s2s_e1.py``, agora generalizado para qualquer dataset que siga o
contrato de ``SpatiotemporalTensorDataset``.
"""

from train_stconv_s2s_e1 import main


if __name__ == "__main__":
    raise SystemExit(main())

