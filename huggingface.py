from huggingface_hub import HfApi

api = HfApi()

def print_clone_urls(repo_list, base_url):
    for repo in repo_list:
        print(f"{repo.id} - {base_url}{repo.id}.git")

# List public repositories for the user "vkapoor"
models = api.list_models(author="vkapoor")
datasets = api.list_datasets(author="vkapoor")
spaces = api.list_spaces(author="vkapoor")

print("\n### Private Repositories for User: vkapoor ###")
print_clone_urls(models, "https://huggingface.co/")
print_clone_urls(datasets, "https://huggingface.co/datasets/")
print_clone_urls(spaces, "https://huggingface.co/spaces/")

# List public repositories for the organization "KapoorLabs-Copenhagen"
models_org = api.list_models(author="KapoorLabs-Copenhagen")
datasets_org = api.list_datasets(author="KapoorLabs-Copenhagen")
spaces_org = api.list_spaces(author="KapoorLabs-Copenhagen")

print("\n### Private Repositories for Organization: KapoorLabs-Copenhagen ###")
print_clone_urls(models_org, "https://huggingface.co/")
print_clone_urls(datasets_org, "https://huggingface.co/datasets/")
print_clone_urls(spaces_org, "https://huggingface.co/spaces/")

