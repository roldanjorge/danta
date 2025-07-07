import typer


app = typer.Typer()


@app.callback()
def callback():
    """
    Danta 
    """


@app.command()
def welcome():
    """
    Danta 
    """
    typer.echo("Welcome to danta")

