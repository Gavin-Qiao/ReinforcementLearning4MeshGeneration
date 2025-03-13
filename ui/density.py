from tkinter import *

class SlowCH_Manager(Canvas):
    """
    Manages a variable number of slow channel massages
    """
    def __init__(self,master=None,**kwargs):
        Canvas.__init__(self,master,**kwargs)
        self.frame = Frame(self)
        self.create_window(0,0,anchor=N+W,window=self.frame)
        self.row = 0
        self.widgets = []
        self.max = 32
        self._init_entries()


    def _init_entries(self):
        """
        initialize the input area with labels and perhaps default values
        """
        label_id  = Label(self.frame, text='message ID').grid(row = self.row, column = 1)
        label_msg = Label(self.frame, text='message data').grid(row = self.row, column = 2)
        self.row += 1


    def add_entry(self):
        """
        Dynamically add entry to GUI until max number of entries is arrived.
        By SENT specification max 32 slow channel messages are allowed.
        """
        if len(self.widgets) >= self.max:
            print('Im full')
        else:
            label = Label(self.frame, text=str(len(self.widgets))).grid(row = self.row, column = 0)
            entry_id = Entry(self.frame)
            entry_id.grid(row = self.row, column = 1)
            entry_data = Entry(self.frame)
            entry_data.grid(row = self.row, column = 2)
            self.row += 1
            self.widgets.append(entry_id)

    def _ypos(self):
        return sum(x.winfo_reqheight() for x in self.widgets)

if __name__ == "__main__":
    root = Tk()

    manager = SlowCH_Manager(root)
    manager.grid(row=0,column=0)

    scroll = Scrollbar(root)
    scroll.grid(row=0,column=1,sticky=N+S)

    manager.config(yscrollcommand = scroll.set)
    scroll.config(command=manager.yview)
    manager.configure(scrollregion = manager.bbox("all"))

    def command():
        manager.add_entry()
        # update scrollregion
        manager.configure(scrollregion = manager.bbox("all"))

    b = Button(root, text = "add entry", command = command)
    b.grid(row=1,column=0)

    root.mainloop()