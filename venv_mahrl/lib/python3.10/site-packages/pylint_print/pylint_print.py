"""Checker for use of print function."""

from __future__ import absolute_import

import astroid

from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

class PrintChecker(BaseChecker):
    """
    Pylint checker for use of the print function.

    When possible, logging should be used instead of print().

    This checker does not cover use of print statements, which are
    handled by E1601: print-statement, and not applicable when
    using Python >= 3.0.
    """
    __implements__ = (IAstroidChecker,)

    name = 'print-function'
    priority = -1
    msgs = {
        'W8201': (
            'Logging should be used instead of the print() function.',
            'print-function',
            'Use logging functionality over the print() function. ' + \
                'This does not cover print statements, which are handled ' + \
                'by E1601: print-statement, and not applicable when using Python >= 3.0.'
        ),
    }

    def visit_call(self, node):
        """
        Visit calls and check for use of the print function.
        """
        if not isinstance(node.func, astroid.Name):
            return
        if list(node.get_children())[0].as_string() == "print":
            self.add_message('print-function', node=node)

def register(linter):
    """
    Required method to register the checker.

    Args:
        linter: Main interface object for Pylint plugins.
    """
    linter.register_checker(PrintChecker(linter))
