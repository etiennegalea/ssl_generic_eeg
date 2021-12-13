import click


@click.group()
def cli():
    pass

@cli.group()
def lunch():
    pass

@cli.group()
def dinner():
    pass

@cli.command()
@click.option('--name', '-n', nargs=2, default=['Etienne', 'Galea'])
def burger(name):
    print(f"Hello world! My name is {name[0]} {name[1]}")


lunch.add_command(burger)
dinner.add_command(burger)

if __name__ == '__main__':
    cli()