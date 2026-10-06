#!/usr/bin/env python3
"""
Manim Static Linter & Sanity Checker
Performs strict static analysis on Manim scene code against the installed package.
Usage:
    python tools/check.py <scene.py>
"""

import ast
import inspect
import sys
import os
import difflib

try:
    import manim
except ImportError:
    print("Error: 'manim' is not installed in the active Python environment.", file=sys.stderr)
    sys.exit(1)


DEPRECATED_APIS = {
    "ShowCreation": "Create",
    "ApplyMethod": "mobject.animate",
    "TextMobject": "Text or Tex",
    "TexMobject": "Tex or MathTex",
    "SmallDot": "Dot(radius=...)",
    "FadeInFrom": "FadeIn(..., shift=...)",
    "FadeInFromDown": "FadeIn(..., shift=DOWN)",
    "FadeOutAndShift": "FadeOut(..., shift=...)",
    "FadeOutAndShiftDown": "FadeOut(..., shift=DOWN)",
    "DashedVMobject": "DashedPath",
    "CONFIG": "class __init__ arguments",
}

INTRODUCING_ANIMATIONS = {
    "Create", "FadeIn", "Write", "DrawBorderThenFill", "GrowFromCenter",
    "GrowFromPoint", "GrowFromEdge", "SpinInFromNothing", "AddTextLetterByLetter",
    "ShowIncreasingSubsets", "FadeInFromPoint", "SpiralIn"
}


class ManimChecker(ast.NodeVisitor):
    def __init__(self, filename, source_code):
        self.filename = filename
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.errors = []
        self.warnings = []
        
        # State tracking
        self.imports_manim_star = False
        self.manim_imported_names = set()
        self.variable_types = {}  # var_name -> class_name
        self.added_mobjects = set()  # set of var_name
        self.manim_symbols = set(dir(manim))
        import builtins
        self.known_local_symbols = set(dir(builtins))

    def error(self, line, msg):
        self.errors.append((line, msg))

    def warning(self, line, msg):
        self.warnings.append((line, msg))

    def visit_ClassDef(self, node):
        self.known_local_symbols.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.known_local_symbols.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.known_local_symbols.add(name)
            if alias.name == "manim":
                self.manim_symbols.add("manim")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "manim":
            for alias in node.names:
                if alias.name == "*":
                    self.imports_manim_star = True
                else:
                    name = alias.asname or alias.name
                    self.manim_imported_names.add(name)
                    if alias.name in DEPRECATED_APIS:
                        self.error(node.lineno, f"Deprecated API imported: '{alias.name}'. Use '{DEPRECATED_APIS[alias.name]}' instead.")
                    elif alias.name not in self.manim_symbols:
                        matches = difflib.get_close_matches(alias.name, self.manim_symbols, n=3, cutoff=0.6)
                        hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
                        self.error(node.lineno, f"Imported name '{alias.name}' not found in manim {manim.__version__}.{hint}")
        else:
            # Non-manim import: record explicitly imported symbols as known local symbols
            for alias in node.names:
                name = alias.asname or alias.name
                self.known_local_symbols.add(name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Track assignments like: circle = Circle(...) or self.circle = Circle(...)
        if isinstance(node.value, ast.Call):
            func_name = self._get_call_func_name(node.value.func)
            if func_name and func_name in self.manim_symbols:
                target_cls = getattr(manim, func_name, None)
                if inspect.isclass(target_cls) and issubclass(target_cls, manim.Mobject):
                    for target in node.targets:
                        var_name = self._get_target_name(target)
                        if var_name:
                            self.variable_types[var_name] = func_name
        self.generic_visit(node)

    def visit_Call(self, node):
        func_name = self._get_call_func_name(node.func)

        # Check for deprecated API calls
        if func_name in DEPRECATED_APIS:
            self.error(node.lineno, f"Deprecated API call: '{func_name}'. Use '{DEPRECATED_APIS[func_name]}' instead.")

        # Check for Animation classes used as methods: e.g. circle.Create()
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            val_name = self._get_target_name(node.func.value)
            if attr_name in self.manim_symbols:
                attr_obj = getattr(manim, attr_name, None)
                if inspect.isclass(attr_obj) and issubclass(attr_obj, manim.Animation):
                    self.error(
                        node.lineno,
                        f"Animation class '{attr_name}' used as method on '{val_name}.{attr_name}()'. "
                        f"Animations are classes passed to self.play({attr_name}({val_name})), not methods."
                    )

            # Check method calls on known variable types: e.g. circle.invalid_method()
            if val_name and val_name in self.variable_types:
                cls_name = self.variable_types[val_name]
                cls_obj = getattr(manim, cls_name, None)
                if cls_obj and inspect.isclass(cls_obj):
                    if not hasattr(cls_obj, attr_name) and attr_name not in {"animate", "copy", "add"}:
                        close = difflib.get_close_matches(attr_name, dir(cls_obj), n=3, cutoff=0.6)
                        hint = f" Did you mean: {', '.join(close)}?" if close else ""
                        self.error(node.lineno, f"Method '{attr_name}' does not exist on {cls_name}.{hint}")
                    elif hasattr(cls_obj, attr_name):
                        # Verify method signature if inspectable
                        method = getattr(cls_obj, attr_name)
                        if callable(method):
                            self._check_call_args(node, method, f"{cls_name}.{attr_name}", is_method=True)

        # Check for direct calls to unknown Manim-style symbols
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if (self.imports_manim_star or name in self.manim_imported_names) and name not in self.manim_symbols and name not in self.known_local_symbols:
                # If looks like class/manim object
                if name[0].isupper() or name in DEPRECATED_APIS:
                    matches = difflib.get_close_matches(name, self.manim_symbols, n=3, cutoff=0.6)
                    hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
                    self.error(node.lineno, f"Unknown Manim symbol '{name}'.{hint}")

        # Check constructor arguments for direct Manim classes
        if func_name and func_name in self.manim_symbols:
            cls_obj = getattr(manim, func_name, None)
            if inspect.isclass(cls_obj):
                self._check_call_args(node, cls_obj, func_name, is_method=False)

        # Track self.add(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            if self._is_self(node.func.value):
                for arg in node.args:
                    arg_name = self._get_target_name(arg)
                    if arg_name:
                        self.added_mobjects.add(arg_name)

        # Track self.play(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "play":
            if self._is_self(node.func.value):
                self._inspect_play_call(node)

        self.generic_visit(node)

    def _inspect_play_call(self, play_node):
        for arg in play_node.args:
            # Check for: self.play(mobj.animate.shift(...))
            if isinstance(arg, ast.Call):
                anim_name = self._get_call_func_name(arg.func)
                
                # Check for: obj.animate.method(...)
                if isinstance(arg.func, ast.Attribute) and isinstance(arg.func.value, ast.Attribute):
                    if arg.func.value.attr == "animate":
                        mobj_name = self._get_target_name(arg.func.value.value)
                        if mobj_name and mobj_name in self.variable_types:
                            if mobj_name not in self.added_mobjects:
                                self.warning(
                                    play_node.lineno,
                                    f"Mobject '{mobj_name}' animated with .animate before being added to scene. "
                                    f"Ensure it is introduced via self.add({mobj_name}) or Create/FadeIn beforehand."
                                )
                
                # Check for standard animations: self.play(Anim(mobj, ...))
                if anim_name:
                    if anim_name in INTRODUCING_ANIMATIONS:
                        for anim_arg in arg.args:
                            m_name = self._get_target_name(anim_arg)
                            if m_name:
                                self.added_mobjects.add(m_name)
                    elif anim_name in {"Transform", "ReplacementTransform"}:
                        if len(arg.args) >= 1:
                            m1_name = self._get_target_name(arg.args[0])
                            if m1_name and m1_name in self.variable_types and m1_name not in self.added_mobjects:
                                self.warning(
                                    play_node.lineno,
                                    f"Transform source '{m1_name}' has not been added to scene before Transform."
                                )
                        if anim_name == "ReplacementTransform" and len(arg.args) >= 2:
                            m2_name = self._get_target_name(arg.args[1])
                            if m2_name:
                                self.added_mobjects.add(m2_name)
                    else:
                        # Non-introducing animation like FadeOut, Rotate, Indicate
                        if len(arg.args) >= 1:
                            m_name = self._get_target_name(arg.args[0])
                            if m_name and m_name in self.variable_types and m_name not in self.added_mobjects:
                                self.warning(
                                    play_node.lineno,
                                    f"Mobject '{m_name}' passed to non-introducing animation '{anim_name}' before being added to scene."
                                )

    def _check_call_args(self, call_node, callable_obj, label, is_method=False):
        try:
            sig = inspect.signature(callable_obj)
        except (ValueError, TypeError):
            return

        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        
        # Check required positional arguments
        req_pos = [
            name for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        
        if is_method and req_pos and req_pos[0] == "self":
            req_pos = req_pos[1:]
            
        num_pos_args = len(call_node.args)
        kwarg_names = {kw.arg for kw in call_node.keywords if kw.arg is not None}
        
        # Missing required args
        for req in req_pos:
            if req not in kwarg_names and num_pos_args == 0 and not has_var_pos:
                self.error(call_node.lineno, f"Call to '{label}' is missing required argument '{req}'.")
                break

    def _is_self(self, node):
        return isinstance(node, ast.Name) and node.id == "self"

    def _get_target_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            val = self._get_target_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None

    def _get_call_func_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


def check_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.", file=sys.stderr)
        return False
        
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
        
    try:
        tree = ast.parse(code, filename=filepath)
    except SyntaxError as e:
        print(f"Syntax Error in {filepath}:{e.lineno}:{e.offset}: {e.msg}")
        return False

    checker = ManimChecker(filepath, code)
    checker.visit(tree)

    print(f"=== Static Analysis Report: {filepath} ===")
    if not checker.errors and not checker.warnings:
        print("  All checks PASSED! Zero errors or warnings detected.")
        return True

    if checker.errors:
        print(f"\n[ERRORS] ({len(checker.errors)}):")
        for line, msg in checker.errors:
            print(f"  Line {line:3d}: {msg}")

    if checker.warnings:
        print(f"\n[WARNINGS] ({len(checker.warnings)}):")
        for line, msg in checker.warnings:
            print(f"  Line {line:3d}: {msg}")

    return len(checker.errors) == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/check.py <scene.py>")
        sys.exit(1)
        
    success = check_file(sys.argv[1])
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
