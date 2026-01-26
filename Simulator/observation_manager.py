class ObservationManager:
    """
    管理哪些資源被觀察，並在資源狀態變化時通知 World
    """

    def __init__(self, on_trigger):
        self.on_trigger = on_trigger
        self.observed_resources = set()

    def add(self, resource_id):
        self.observed_resources.add(resource_id)

    def remove(self, resource_id):
        if resource_id in self.observed_resources:
            print("observation remove resource")
            self.observed_resources.remove(resource_id)
            # 👇 關鍵：被移除時觸發
            self.on_trigger()
