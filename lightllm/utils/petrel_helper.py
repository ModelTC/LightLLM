import os
import io
import torch
import json
import configparser
import pickle as pk
import os

class PetrelOpen(object):
    def __init__(self, filename, **kwargs):
        self.handle = PetrelHelper._petrel_helper.load_data(filename, **kwargs)

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc_value, exc_trackback):
        del self.handle


class PetrelHelper(object):

    _petrel_helper = None
    open = PetrelOpen

    default_conf_path = os.environ.get('PETRELPATH', '~/petreloss.conf')

    def __init__(self, conf_path=default_conf_path):
        self.conf_path = conf_path
        self._inited = False
        self._init_petrel()

        PetrelHelper._petrel_helper = self

    def _init_petrel(self):
        try:
            from petrel_client.client import Client
            self.client = Client(self.conf_path)

            self._inited = True
        except:
            pass

    def check_init(self):
        if not self._inited:
            raise Exception('petrel oss not inited')

    def _iter_cpeh_lines(self, path):
        response = self.client.get(path, enable_stream=True, no_cache=True)

        for line in response.iter_lines():
            cur_line = line.decode('utf-8')
            yield cur_line
    
    def load_data(self, path, ceph_read=True, fs_read=False, mode='r'):
        if 's3://' not in path:
            if not fs_read:
                return open(path, mode)
            else:
                return open(path, mode).read()
        else:
            self.check_init()

            if ceph_read:
                return self._iter_cpeh_lines(path)
            else:
                return self.client.get(path)

    @staticmethod
    def load_pk(path, mode='r'):
        if 's3://' not in path:
            pk_res = pk.load(open(path, mode))
        else:
            pk_res = pk.loads(PetrelHelper._petrel_helper.load_data(path, ceph_read=False))
        return pk_res

    @staticmethod
    def load_json(path, mode='r'):
        if 's3://' not in path:
            js = json.load(open(path, mode))
        else:
            js = json.loads(PetrelHelper._petrel_helper.load_data(path, ceph_read=False))
        return js

    @staticmethod
    def download_json(path, local_path, mode='r'):
        if 's3://' not in path:
            js = json.load(open(path, mode))
        else:
            js = json.loads(PetrelHelper._petrel_helper.load_data(path, ceph_read=False))
        
        with open(local_path, 'w') as f:
            json.dump(js, f)
   
    @staticmethod
    def download_file(path, local_path, mode='r'):
        data = PetrelHelper._petrel_helper.load_data(path, ceph_read=False)
        with open(local_path, 'wb') as f:
            f.write(data)


    def list(path, extension="json"):
        if 's3://' not in path:
            assert os.path.exists(path), f'No such file: {path}'
            filenames = []
            for name in os.listdir(path):
                if name.endswith(extension) or extension == "all":
                    filenames.append(name)
            return filenames
        else:
            contents = PetrelHelper._petrel_helper.client.list(path)
            filenames = []
            for name in contents:
                if name.endswith(extension) or extension == "all":
                    filenames.append(name)
        return filenames

    def load_pretrain(self, path, map_location=None):
        if 's3://' not in path:
            assert os.path.exists(path), f'No such file: {path}'
            return torch.load(path, map_location=map_location)
        elif 'http://' in path:
            return torch.hub.load_state_dict_from_url(path, map_location=map_location)
        else:
            self.check_init()

            file_bytes = self.client.get(path)
            buffer = io.BytesIO(file_bytes)
            res = torch.load(buffer, map_location=map_location)
            return res

    @staticmethod
    def load(path, **kwargs):
        if '.ini' in path:
            path = path[:-4]
        if not os.path.exists(path) and os.path.exists(path + '.ini'):
            # get realpath
            conf = configparser.ConfigParser()
            conf.read(path + '.ini')
            path = conf['Link']['ceph']
        return PetrelHelper._petrel_helper.load_pretrain(path, **kwargs)

    def save_checkpoint(self, model, path):
        if 's3://' not in path:
            torch.save(model, path)
        else:
            with io.BytesIO() as f:
                torch.save(model, f)
                f.seek(0)
                self.client.put(path, f)

    @staticmethod
    def get_stream(path):
        if "s3://" in path:
            stream = io.BytesIO(PetrelHelper._petrel_helper.client.get(path))
        else:
            stream = open(path, "rb")
        return stream

    @staticmethod
    def exists(path):
        if "s3://" in path:
            return PetrelHelper._petrel_helper.client.contains(path)
        else:
            return os.path.exists(path)

    @staticmethod
    def save(model, path, ceph_path=None):
        if ceph_path:
            # save link
            lustre_path = os.path.abspath(path)

            link_path = path + '.ini'
            config = configparser.ConfigParser()
            config.add_section("Link")
            config.set("Link", "path", path)
            config.set("Link", "lustre", lustre_path)
            config.set("Link", "ceph", ceph_path)

            # save model to ceph_path
            ret = PetrelHelper._petrel_helper.save_checkpoint(model, ceph_path)

            # save model before saving ini
            config.write(open(link_path, "w"))
            return ret
        else:
            return PetrelHelper._petrel_helper.save_checkpoint(model, path)

__petrel_helper = PetrelHelper()

def s3_model_prepare(ceph_file_path: str):
    """
    Create a temporary directory to download JSON and tokenizer files locally for easy access.
    """
    assert 's3://' in ceph_file_path
    if not os.path.exists(ceph_file_path):
        os.makedirs(ceph_file_path) 

    json_files = PetrelHelper.list(ceph_file_path, extension='json')
    tokenizer_model = PetrelHelper.list(ceph_file_path, extension='model')

    if tokenizer_model and (not os.path.exists(os.path.join(ceph_file_path, tokenizer_model[0]))):
        PetrelHelper.download_file(os.path.join(ceph_file_path, tokenizer_model[0]), os.path.join(ceph_file_path, tokenizer_model[0]))
    for json_file in json_files:
        if not os.path.exists(os.path.join(ceph_file_path, json_file)):
            PetrelHelper.download_json(os.path.join(ceph_file_path, json_file), os.path.join(ceph_file_path, json_file))
    return

def s3_model_clear(ceph_file_path: str):
    """
    Once all files are read, the temporary directory is deleted.
    """
    os.rmdir(ceph_file_path)
