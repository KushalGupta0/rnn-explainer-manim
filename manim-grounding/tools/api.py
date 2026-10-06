#!/usr/bin/env python3
"""
Manim API Lookup Tool (Grounding)
Source of truth: Currently installed manim package.
Usage:
    python tools/api.py <Name>
    python tools/api.py --list-animations
    python tools/api.py --methods <Class>
"""

import sys
import inspect
import difflib
import argparse

try:
    import manim
except ImportError:
    print("Error: 'manim' is not installed in the active Python environment.", file=sys.stderr)
    sys.exit(1)


def get_kind(obj):
    if inspect.isclass(obj):
        if issubclass(obj, manim.VMobject):
            return "VMobject"
        if issubclass(obj, manim.Mobject):
            return "Mobject"
        if issubclass(obj, manim.Animation):
            return "Animation"
        if issubclass(obj, manim.Scene):
            return "Scene"
        return "Class"
    if callable(obj):
        mod = getattr(obj, "__module__", "")
        if "rate_functions" in mod:
            return "rate function"
        return "function"
    return "constant"


def format_signature(obj):
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except Exception:
        return "(...)"


def list_animations():
    print(f"=== Installed Manim Animations (Manim {manim.__version__}) ===")
    anims = []
    for name in dir(manim):
        if name.startswith("_"):
            continue
        obj = getattr(manim, name)
        if inspect.isclass(obj) and issubclass(obj, manim.Animation):
            anims.append((name, obj))
    anims = sorted(anims, key=lambda x: x[0])
    for name, cls in anims:
        sig = format_signature(cls)
        doc = (inspect.getdoc(cls) or "").split("\n")[0]
        doc_snippet = f" - {doc}" if doc else ""
        print(f"• {name}{sig}{doc_snippet}")
    print(f"\nTotal Animation classes: {len(anims)}")


def list_methods(cls_name):
    if not hasattr(manim, cls_name):
        matches = difflib.get_close_matches(cls_name, dir(manim), n=5, cutoff=0.5)
        print(f"Error: Class '{cls_name}' not found in manim.")
        if matches:
            print("Did you mean: " + ", ".join(matches))
        sys.exit(1)
        
    cls = getattr(manim, cls_name)
    if not inspect.isclass(cls):
        print(f"Error: '{cls_name}' is not a class (kind: {get_kind(cls)}).")
        sys.exit(1)
        
    print(f"=== Public Methods for {cls_name} ({get_kind(cls)}) ===")
    own_methods = []
    inherited_methods = []
    
    # Collect all callable attributes
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            val = getattr(cls, name)
            if callable(val):
                sig = format_signature(val)
                # Check if defined directly on this class
                if name in cls.__dict__:
                    own_methods.append((name, sig))
                else:
                    inherited_methods.append((name, sig))
        except Exception:
            continue
            
    if own_methods:
        print("\n[Directly Defined / Overridden Methods]")
        for m_name, sig in sorted(own_methods):
            print(f"  {m_name}{sig}")
    else:
        print("\n[Directly Defined Methods]: None (all inherited)")
        
    if inherited_methods:
        print(f"\n[Inherited Public Methods ({len(inherited_methods)} total)]")
        for m_name, sig in sorted(inherited_methods):
            print(f"  {m_name}{sig}")


def inspect_symbol(name):
    if not hasattr(manim, name):
        matches = difflib.get_close_matches(name, dir(manim), n=5, cutoff=0.5)
        print(f"Error: Symbol '{name}' not found in manim {manim.__version__}.")
        if matches:
            print("Close matches found in installed package:")
            for m in matches:
                m_obj = getattr(manim, m)
                print(f"  • {m} ({get_kind(m_obj)})")
        else:
            print("No close matches found.")
        sys.exit(1)

    obj = getattr(manim, name)
    kind = get_kind(obj)
    mod = getattr(obj, "__module__", "manim")
    
    print(f"==================================================")
    print(f" Symbol:       {name}")
    print(f" Kind:         {kind}")
    print(f" Manim Version:{manim.__version__}")
    print(f" Import:       from manim import {name}")
    print(f" Module:       {mod}")
    
    if inspect.isclass(obj):
        sig = format_signature(obj)
        print(f" Signature:    {name}{sig}")
        bases = [b.__name__ for b in obj.__mro__[1:] if b is not object]
        print(f" Parent Bases: {' -> '.join(bases)}")
        
        # Public methods directly on this class
        own_methods = []
        for m_name, val in obj.__dict__.items():
            if not m_name.startswith("_") and callable(val):
                own_methods.append((m_name, format_signature(val)))
        own_methods.sort(key=lambda x: x[0])
        
        print(f"\n Own Public Methods ({len(own_methods)}):")
        if own_methods:
            for m_name, m_sig in own_methods:
                print(f"   • {m_name}{m_sig}")
        else:
            print("   (Inherits all methods from parent classes)")
            
        doc = inspect.getdoc(obj)
        if doc:
            first_line = doc.split("\n")[0]
            print(f"\n Summary:      {first_line}")
            
    elif callable(obj):
        sig = format_signature(obj)
        print(f" Signature:    {name}{sig}")
        doc = inspect.getdoc(obj)
        if doc:
            first_line = doc.split("\n")[0]
            print(f"\n Summary:      {first_line}")
            
    else:
        # Constant / ndarray / primitive
        val_repr = repr(obj)
        if len(val_repr) > 100:
            val_repr = val_repr[:97] + "..."
        print(f" Value:        {val_repr}")
        print(f" Type:         {type(obj).__name__}")
        
    print(f"==================================================")


def main():
    parser = argparse.ArgumentParser(description="Manim API Grounding & Inspection Tool")
    parser.add_argument("name", nargs="?", help="Name of Manim class, function, or constant to inspect")
    parser.add_argument("--list-animations", action="store_true", help="List all available Animation classes")
    parser.add_argument("--methods", metavar="CLASS", help="List all public methods and signatures for a class")
    
    args = parser.parse_args()
    
    if args.list_animations:
        list_animations()
    elif args.methods:
        list_methods(args.methods)
    elif args.name:
        inspect_symbol(args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
