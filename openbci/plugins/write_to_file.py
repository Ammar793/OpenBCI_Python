import plugin_interface as plugintypes


class PluginPrint(plugintypes.IPluginExtended):
    def activate(self):
        print("write_to_file activated")

    # called with each new sample
    def __call__(self, sample):
        if sample:
            sample_string = "%s\n" % (
            str(sample.channel_data)[1:-1])
            f = open("./output.txt", "a+")
            f.write(sample_string)
            f.close()
