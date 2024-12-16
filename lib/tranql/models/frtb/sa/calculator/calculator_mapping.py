from .delta_calculators import CMDeltaCalculator, CSRDeltaCalculator, CSRSecDeltaCalculator
from .delta_calculators import EQDeltaCalculator, FXDeltaCalculator, IRDeltaCalculator
from .vega_calculators import CMVegaCalculator, CSRVegaCalculator, CSRSecVegaCalculator
from .vega_calculators import EQVegaCalculator, FXVegaCalculator, IRVegaCalculator
from .curvature_calculators import CMCurvatureCalculator, CSRCurvatureCalculator, CSRSecCurvatureCalculator
from .curvature_calculators import EQCurvatureCalculator, FXCurvatureCalculator, IRCurvatureCalculator
from .drc_calculator import DRCNonSecCalculator, DRCSecCalculator
from .rrao_calculator import RRAOCalculator

calculator_mapping = {
    'GIRR Delta': IRDeltaCalculator,
    'GIRR Vega': IRVegaCalculator,
    'GIRR Curvature': IRCurvatureCalculator,
    'Equity Delta': EQDeltaCalculator,
    'Equity Vega': EQVegaCalculator,
    'Equity Curvature': EQCurvatureCalculator,
    'Commodity Delta': CMDeltaCalculator,
    'Commodity Vega': CMVegaCalculator,
    'Commodity Curvature': CMCurvatureCalculator,
    'FX Delta': FXDeltaCalculator,
    'FX Vega': FXVegaCalculator,
    'FX Curvature': FXCurvatureCalculator,
    'CSR Non Sec Delta': CSRDeltaCalculator,
    'CSR Non Sec Vega': CSRVegaCalculator,
    'CSR Non Sec Curvature': CSRCurvatureCalculator,
    'CSR Sec non-CTP Delta': CSRSecDeltaCalculator,
    'CSR Sec non-CTP Vega': CSRSecVegaCalculator,
    'CSR Sec non-CTP Curvature': CSRSecCurvatureCalculator,
    'DRC Non-Sec': DRCNonSecCalculator,
    'DRC Sec non-CTP': DRCSecCalculator,
    'RRAO': RRAOCalculator,
}
