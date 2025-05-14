import hydra
from omegaconf import OmegaConf

from data.get_uci import (
    PoleteleDataset,
    ElevatorsDataset,
    BikeDataset,
    Kin40KDataset,
    ProteinDataset,
    KeggDirectedDataset,
    CTSlicesDataset,
    KeggUndirectedDataset,
    RoadDataset,
    SongDataset,
    BuzzDataset,
    HouseElectricDataset,
)

from data.synthetic.synthetic import (
    BraninDataset,
    SixHumpCamelDataset,
    StyblinskiTangDataset,
    HartmannDataset,
    WelchDataset,
)
from data.get_md22 import (
    MD22_AcAla3NHME_Dataset,
    MD22_DHA_Dataset,
    MD22_DNA_AT_AT_CG_CG_Dataset,
    MD22_DNA_AT_AT_Dataset,
    MD22_Stachyose_Dataset,
    MD22_Buckyball_Catcher_Dataset,
    MD22_DoubleWalledNanotube_Dataset,
)

import gp.softki.train
import gp.svgp
import gp.svgp.train
import gp.sgpr.train
from gp.util import *


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cli_config):
    OmegaConf.set_struct(cli_config, False)
    print(cli_config)

    # Config and train function factory
    if cli_config.model == "softki":
        train_gp = gp.softki.train.train_gp
        config = cli_config.gp.softki
    elif cli_config.model == "svgp":
        train_gp = gp.svgp.train.train_gp
        config = cli_config.gp.svgp
    elif cli_config.model == "sgpr":
        train_gp = gp.sgpr.train.train_gp
        config = cli_config.gp.sgpr
    else:
        raise ValueError(f"Name not found {config.model.name}")
    
    config = OmegaConf.merge(config, {
        "data_dir": cli_config.data_dir,
        "dataset": cli_config.dataset,
        "wandb": cli_config.wandb,
        "synthetic": cli_config.synthetic,
        "draw": cli_config.draw,
    })

    get_grad = True

    UCI = ["pol", "elevators", "bike", "kin40k", "protein", "keggdirected", "slice", "keggundirected", "3droad", "song", "buzz", "houseelectric"]
    if config.dataset.name in UCI:
        assert cli_config.model not in ["dsoftki", "ddsvgp", "dsvgp", "dexact_gp"]

    # Dataset factory
    if config.dataset.name == "pol":
        dataset = PoleteleDataset(f"{config.data_dir}/pol/data.csv")
    elif config.dataset.name == "elevators":
        dataset = ElevatorsDataset(f"{config.data_dir}/elevators/data.csv")
    elif config.dataset.name == "bike":
        dataset = BikeDataset(f"{config.data_dir}/bike/data.csv")
    elif config.dataset.name == "kin40k":
        dataset = Kin40KDataset(f"{config.data_dir}/kin40k/data.csv")
    elif config.dataset.name == "protein":
        dataset = ProteinDataset(f"{config.data_dir}/protein/data.csv")
    elif config.dataset.name == "keggdirected":
        dataset = KeggDirectedDataset(f"{config.data_dir}/keggdirected/data.csv")
    elif config.dataset.name == "slice":
        dataset = CTSlicesDataset(f"{config.data_dir}/slice/data.csv")
    elif config.dataset.name == "keggundirected":
        dataset = KeggUndirectedDataset(f"{config.data_dir}/keggundirected/data.csv")
    elif config.dataset.name == "3droad":
        dataset = RoadDataset(f"{config.data_dir}/3droad/data.csv")
    elif config.dataset.name == "song":
        dataset = SongDataset(f"{config.data_dir}/song/data.csv")
    elif config.dataset.name == "buzz":
        dataset = BuzzDataset(f"{config.data_dir}/buzz/data.csv")
    elif config.dataset.name == "houseelectric":
        dataset = HouseElectricDataset(f"{config.data_dir}/houseelectric/data.csv")
    
    elif config.dataset.name == "branin":
        dataset = BraninDataset(config.synthetic.N)
    elif config.dataset.name == "six-hump-camel":
        dataset = SixHumpCamelDataset(config.synthetic.N)
    elif config.dataset.name == "styblinski-tang":
        dataset = StyblinskiTangDataset(config.synthetic.N)
    elif config.dataset.name == "hartmann":
        dataset = HartmannDataset(config.synthetic.N)
    elif config.dataset.name == "welch":
        dataset = WelchDataset(config.synthetic.N)

    elif config.dataset.name == "Ac-Ala3-NHMe":
        dataset = MD22_AcAla3NHME_Dataset(f"{config.data_dir}/md22_Ac-Ala3-NHMe.npz", get_forces=get_grad)
    elif config.dataset.name == "AT-AT":
        dataset = MD22_DNA_AT_AT_Dataset(f"{config.data_dir}/md22_AT-AT.npz", get_forces=get_grad)
    elif config.dataset.name == "AT-AT-CG-CG":
        dataset = MD22_DNA_AT_AT_CG_CG_Dataset(f"{config.data_dir}/md22_AT-AT-CG-CG.npz", get_forces=get_grad)
    elif config.dataset.name == "stachyose":
        dataset = MD22_Stachyose_Dataset(f"{config.data_dir}/md22_stachyose.npz", get_forces=get_grad)
    elif config.dataset.name == "DHA":
        dataset = MD22_DHA_Dataset(f"{config.data_dir}/md22_DHA.npz", get_forces=get_grad)
    elif config.dataset.name == "buckyball-catcher":
        dataset = MD22_Buckyball_Catcher_Dataset(f"{config.data_dir}/md22_buckyball-catcher.npz", get_forces=get_grad)
    elif config.dataset.name == "double-walled-nanotube":
        dataset = MD22_DoubleWalledNanotube_Dataset(f"{config.data_dir}/md22_double-walled_nanotube.npz", get_forces=get_grad)

    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported ...")
    
    # Seed
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    # Generate splits
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_frac=config.dataset.train_frac,
        val_frac=config.dataset.val_frac
    )

    # Train
    if config.dataset.name in UCI:
        model = train_gp(config, train_dataset, test_dataset, collate_fn=None)
    else:
        model = train_gp(config, train_dataset, test_dataset)

    # Optional draw
    if config.draw:
        import matplotlib.pyplot as plt
        from data.synthetic.synthetic import from_unit_cube, mk_synthetic
        from data.synthetic.synthetic_functions import Branin, SixHumpCamel, StyblinskiTang
        
        draw = True
        if config.dataset.name == "branin":
            X, Y, Z, lb, ub = mk_synthetic(Branin())
        elif config.dataset.name == "six-hump-camel":
            X, Y, Z, lb, ub = mk_synthetic(SixHumpCamel())
        elif config.dataset.name == "styblinski-tang":
            X, Y, Z, lb, ub = mk_synthetic(StyblinskiTang())
        else:
            draw = False
        
        if draw and isinstance(model, gp.dsoft_ki.model.DSoftKI):
            print("Drawing ...")
            # Create plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            
            # Plot original
            surface = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none', alpha=0.7)
            
            # Plot test
            xs = torch.stack([x for x, y in test_dataset])
            pred_ys = model.pred(xs.to(config.model.device))[:len(xs)].detach().cpu().numpy()
            scaled_xs = from_unit_cube(xs, lb, ub)
            x1 = np.array([x[0] for x in scaled_xs])
            x2 = np.array([x[1] for x in scaled_xs])
            ax.scatter(x1, x2, pred_ys, c='red', s=0.1)

            fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
            fig.savefig(f"./analysis/media/synthetic_{cli_config.model}_{config.dataset.name}_{config.training.seed}.png")


if __name__ == "__main__":
    main()
