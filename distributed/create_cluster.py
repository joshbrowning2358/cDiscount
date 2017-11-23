import subprocess
import shlex


class DistributedTensorflowExecutor(object):
    def __init__(self, num_workers, num_masters=1, num_parameter_servers=1, type='cpu',
                 master_machine_type='n1-standard-2',
                 worker_machine_type='n1-standard-4',
                 ps_machine_type='n1-standard-2'):
        self.num_workers = num_workers
        self.num_masters = num_masters
        self.num_parameter_servers = num_parameter_servers
        if type in ['cpu', 'gpu']:
            self.type = type
        else:
            raise ValueError('Unsupported type {}, must be cpu or gpu'.format(type))
        self.master_machine_type = master_machine_type
        self.worker_machine_type = worker_machine_type
        self.ps_machine_type = ps_machine_type

    def initialize_cluster(self):
        base_command = 'gcloud compute instances create {machines} ' +\
            '--image template-image --machine-type {machine_type} ' +\
            '--scopes default,storage-rw'
        master = base_command.format(machines=' '.join(['master-' + str(i) for i in range(self.num_masters)]),
                                     machine_type=self.master_machine_type)
        worker = base_command.format(machines=' '.join(['worker-' + str(i) for i in range(self.num_workers)]),
                                     machine_type=self.worker_machine_type)
        ps = base_command.format(machines=' '.join(['ps-' + str(i) for i in range(self.num_parameter_servers)]),
                                 machine_type=self.ps_machine_type)
        self._run_on_commandline(master)
        self._run_on_commandline(worker)
        self._run_on_commandline(ps)

    @staticmethod
    def _run_on_commandline(command):
        """
        Executes "command" on commandline -
            Returns the output on successful execution
            Raises error when the command returns a non-zero exit code
        """
        print('\n\n{}'.format('=' * 100))
        print('EXECUTING COMMANDLINE')
        print('=' * 100)
        print('\n{}\n'.format(command))
        args = shlex.split(command)
        try:
            output = subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise Exception('"{}" failed: {}'.format(command, error.output.decode('utf-8')))
        print('=' * 100)
        print('EXECUTION COMPLETED')
        print('{}\n\n'.format('=' * 100))
        output = output.decode('utf-8')
        return output
