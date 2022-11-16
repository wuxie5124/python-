from tkinter import Widget
from os import path
import sys
from tkinter import *
class Ribbon(Widget):
    def __init__(self, master, kw=None):
        self.version = master.tk.call('package','require','tkribbon')
        self.library = master.tk.eval('set ::tkribbon::library')
        Widget.__init__(self, master, 'tkribbon::ribbon', kw=kw)
    def load_resource(self, resource_file, resource_name='APPLICATION_RIBBON'):
        """Load the ribbon definition from resources.
        Ribbon markup is compiled using the uicc compiler and the resource included
        in a dll. Load from the provided file."""
        self.tk.call(self._w, 'load_resources', resource_file)
        self.tk.call(self._w, 'load_ui', resource_file, resource_name)
if __name__ == '__main__':

    def main():
        root = Tk()
        r = Ribbon(root)
        name = 'APPLICATION_RIBBON'
        if len(sys.argv) > 1:
            resource = sys.argv[1]
        if len(sys.argv) > 2:
            name = sys.argv[2]
        else:
            resource = path.join(r.library, 'libtkribbon1.0.dll')
            r.load_resource(resource, name)
        t = Text(root)
        r.grid(sticky=(N,E,S,W))
        t.grid(sticky=(N,E,S,W))
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.mainloop()
    main()