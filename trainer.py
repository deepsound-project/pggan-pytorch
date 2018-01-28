import heapq


# Based on torch.utils.trainer.Trainer code.
class Trainer(object):

    def __init__(self,
                 D,
                 G,
                 D_loss,
                 G_loss,
                 optimizer_d,
                 optimizer_g,
                 dataset,
                 dataiter,
                 random_latents_generator,
                 D_training_repeats=1,  # trainer
                 tick_nimg_default=2 * 1000,  # trainer
                 resume_nimg=0):
        self.D = D
        self.G = G
        self.D_loss = D_loss
        self.G_loss = G_loss
        self.D_training_repeats = D_training_repeats
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g
        self.dataiter = dataiter
        self.dataset = dataset
        self.cur_nimg = resume_nimg
        self.random_latents_generator = random_latents_generator
        self.tick_start_nimg = self.cur_nimg
        self.tick_duration_nimg = tick_nimg_default
        self.iterations = 0
        self.cur_tick = 0
        self.time = 0
        self.stats = {
            'kimg_stat': { 'val': self.cur_nimg / 1000., 'log_epoch_fields': ['{val:8.3f}'], 'log_name': 'kimg' },
            'tick_stat': { 'val': self.cur_tick, 'log_epoch_fields': ['{val:5}'], 'log_name': 'tick'}
        }
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            's':    []
        }

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for (duration, unit) in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self, total_kimg=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        while self.cur_nimg < total_kimg * 1000:
            self.train()
            if self.cur_nimg >= self.tick_start_nimg + self.tick_duration_nimg or self.cur_nimg >= total_kimg * 1000:
                self.cur_tick += 1
                self.tick_start_nimg = self.cur_nimg
                self.stats['kimg_stat']['val'] = self.cur_nimg / 1000.
                self.stats['tick_stat']['val'] = self.cur_tick
                self.call_plugins('epoch', self.cur_tick)

    def train(self):
        real_images_expr = next(self.dataiter).cuda()
        fake_latents_in = self.random_latents_generator().cuda()

        # Calculate loss and optimize
        d_losses = [0, 0, 0]
        for i in range(self.D_training_repeats):
            d_losses = self.D_loss(self.D, self.G, real_images_expr, fake_latents_in)
            d_losses = tuple(d_losses)
            D_loss = d_losses[0]
            D_loss.backward()
            self.optimizer_d.step()

        g_losses = self.G_loss(self.G, self.D, fake_latents_in)
        if type(g_losses) is list:
            g_losses = tuple(g_losses)
        elif type(g_losses) is not tuple:
            g_losses = (g_losses,)
        G_loss = g_losses[0]
        G_loss.backward()
        self.optimizer_g.step()

        # tick_train_out.append((G_loss, D_loss, D_real, D_fake))
        self.cur_nimg += real_images_expr.size(0)
        self.iterations += 1
        self.call_plugins('iteration', self.iterations, *(g_losses + d_losses))

