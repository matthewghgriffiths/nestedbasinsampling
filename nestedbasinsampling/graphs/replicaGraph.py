# -*- coding: utf-8 -*-

from itertools import chain, izip, groupby
import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

from pele.utils.disconnectivity_graph import DisconnectivityGraph

from nestedbasinsampling.storage import (
    Minimum, Replica, Run, Path, TransitionState, Database)
from nestedbasinsampling.sampling.stats import AndersonDarling, CDF
from nestedbasinsampling.nestedsampling.combine import combineAllRuns
from nestedbasinsampling.utils import (
    dict_update_copy, iter_minlength, len_iter, GraphError)
from nestedbasinsampling.utils.sortedcollection import SortedCollection

from functools import wraps
def wrap_output(fn=None, wrapper=list):
    """
    A decorator which wraps a function's return value in ``list(...)``.

    Useful when an algorithm can be expressed more cleanly as a generator but
    the function should return an list.

    Example::

        >>> @wrap_output
        ... def get_lengths(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths(["spam", "eggs"])
        [4, 4]
        >>>
        >>> @wrap_output(wrapper=tuple)
        ... def get_lengths_tuple(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths_tuple(["foo", "bar"])
        (3, 3)
    """
    def wrap_return(fn):
        @wraps(fn)
        def wrap_helper(*args, **kw):
            return wrapper(fn(*args, **kw))
        return wrap_helper
    if fn is None:
        return wrap_return
    return wrap_return(fn)
    
class ReplicaGraph(object):
    """This object wraps around the database that the results of the
    nested basin-sampling are stored in and a directed networkx
    graph to allow more straightforward interaction with the results.

    It is recommended to interact with the database through this class

    Attributes
    ----------
    database : Database
        The sqlalchemy database where the Replicas, Runs and Minimum
        are stored
    graph : networkx.DiGraph
        The directed graph that stores the relationships between the
        replicas. The nodes of the graph are replicas or minima, and
        the edges are nested optimisation runs or associations between
        replicas and minima

    Methods
    -------
    Minimum(energy, coords)
        Creates a new minimum and adds it to the database and graph
    Replica(energy, coords, minimum=None, stepsize=None)
        Creates a new replica and adds it to the database and graph
        can optionally associate it with a minimum or a stepsize
    Run(Emax, nlive, parent, child,
        volume=1., configs=None, stepsizes=None)
        Creates a new nested run and adds it to the database and graph
    """

    def __init__(self, database=None, run_adder=None, rep_adder=None,
                 min_adder=None, path_adder=None, Etol=1e-5):

        self.Etol = Etol

        database = Database() if database is None else database
        # These methods will be used to create new objects
        self.NewRun = database.addRun     if run_adder is None else run_adder
        self.NewRep = database.addReplica if rep_adder is None else rep_adder
        self.NewMin = database.addMinimum if min_adder is None else min_adder
        self.NewPath = database.addPath if path_adder is None else path_adder

        self.on_minimum_added = []
        self.on_run_added = []
        self.on_replica_added = []
        self.on_path_added = []


        self.Emax = None
        self.disconnect_kw = {}

        # Create new graph from database
        self.loadFromDatabase(database, newGraph=True)

    def loadFromDatabase(self, database, newGraph=True):
        """
        """
        if newGraph:
            self.graph = nx.MultiDiGraph()

        self.database = database
        self.runs = self.database.runs
        self.paths = self.database.paths
        self.replicas = self.database.replicas
        self.minima = self.database.minima

        [self.addMinimum(m) for m in self.database.minima()]
        [self.addReplica(rep) for rep in self.database.replicas()]
        [self.addRun(run) for run in self.database.runs()]
        [self.addPath(path) for path in self.database.paths()]

    def Minimum(self, energy, coords):
        m = self.NewMin(energy, coords)
        for event in self.on_minimum_added:
            event(m)
        self.addMinimum(m)
        return m

    def Replica(self, energy, coords, stepsize=None):
        rep = self.NewRep(energy, coords, stepsize=stepsize)
        for event in self.on_replica_added:
            event(rep)
        self.addReplica(rep)
        return rep

    def Run(self, Emax, nlive, parent, child, intermediates=[],
            stored=None, configs=None, stepsizes=None):
        """
        """
        run = self.NewRun(Emax, nlive, parent, child,
                          stored=stored, configs=configs,
                          stepsizes=stepsizes)
        for event in self.on_run_added:
            event(run)
        self.addRun(run)
        return run

    def Path(self, energy, parent, child, intermediates=[], energies=None,
             stored=None, configs=None):
        """
        """
        path = self.NewPath(
            energy, parent, child, energies=energies,
            stored=stored, configs=configs, quench=quench, minimum=minimum)
        for event in self.on_path_added:
            event(path)
        self.addPath(path)
        return path

    def addMinimum(self, m):
        """
        """
        self.graph.add_node(m, energy=m.energy)

    def addReplica(self, replica):
        """
        """
        self.graph.add_node(replica, energy=replica.energy)

    def addPath(self, path, **kwargs):
        """Adds path to graph will connect parent to child
        through intermediates"""
        self.graph.add_edge(
            path.parent, path.child, path=path, **kwargs)

    def addRun(self, run, **kwargs):
        """Adds run to graph will connect parent to child
        through intermediates"""
        self.graph.add_edge(
            run.parent, run.child, run=run, **kwargs)

    def basin_runs_iter(self, minimum):
        """
        """
        return (
            edge['run'] 
            for parent, m, edge in self.graph.in_edges_iter(minimum, True)
            if edge.has_key('run'))
                
    basin_runs = wrap_output(basin_runs_iter)
    
    def basin_run(self, minimum):
        return combineAllRuns(self.basin_runs(minimum))
    
    def nested_runs_iter(self, minimum):
        """
        """
        replicas = (
            parent
            for parent, m, edge in self.graph.in_edges_iter(minimum, True)
            if edge.has_key('path'))
        return (
            edge['run'] 
            for replica in replicas
            for parent, _, edge in self.graph.in_edges_iter(replica, True)
            if edge.has_key('run'))
            
    nested_runs = wrap_output(nested_runs_iter)
    
    def nested_run(self, minima):
        """
        """
        if isinstance(minima, Minimum):
            minima = [minima]
        return combineAllRuns([
            r for m in minima for r in self.nested_runs(m)])



















      
            
